"""
Requires the following installations:

1. R (R studio is the easiest option: https://www.rstudio.com/products/rstudio/download/).
Installing R packages is easy, in R studio, command install.packages("package_name") works
(https://www.dummies.com/programming/r/how-to-install-load-and-unload-packages-in-r/)
The following R packages are required:
    1a. RcppCNPy
    1b. DiceKriging
    1c. GPareto

2. rpy2 -- which runs R under the hood (pip install rpy2 should work)
# http://rpy.sourceforge.net/rpy2/doc-2.1/html/index.html

3. astropy for reading fits files

"""


##### Generic packages ###############
import numpy as np
import matplotlib.pylab as plt
import time

###### astropy for fits reading #######
from astropy.io import fits as pf
import astropy.table

###### R kernel imports from rpy2 #####
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r
from rpy2.robjects.packages import importr


############################# PARAMETERS ##############################

fitsfileIn = "../P_data/2ndpass_vals_for_test.fits"   ## Input fits file
nRankMax = 32   ## Number of basis vectors in truncated PCA
GPmodel = '"R_GP_model7' + str(nRankMax) + '.RData"'  ## Double and single quotes are necessary

################################# I/O #################################
RcppCNPy = importr('RcppCNPy')
# RcppCNPy.chooseCRANmirror(ind=1) # select the first mirror in the list


### 4 parameters: RHO, SIGMA, TAU, SSPT ###

Allfits = pf.open(fitsfileIn)
AllData = astropy.table.Table(Allfits[1].data)

parameter_array = np.array([AllData['RHO'], AllData['SIGMA_LAMBDA'], AllData['TAU'],
                            AllData['SSPT']]).T


def plot_params(lhd):
    f, a = plt.subplots(lhd.shape[1], lhd.shape[1], sharex=True, sharey=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.rcParams.update({'font.size': 8})

    for i in range(lhd.shape[1]):
        for j in range(i + 1):
            print(i, j)
            if (i != j):
                a[i, j].scatter(lhd[:, i], lhd[:, j], s=5)
                a[i, j].grid(True)
            else:
                # a[i,i].set_title(AllLabels[i])
                # a[i, i].text(0.4, 0.4, AllLabels[i], size = 'xx-large')
                hist, bin_edges = np.histogram(lhd[:, i], density=True, bins=64)
                # a[i,i].bar(hist)
                a[i, i].bar(bin_edges[:-1], hist / hist.max(), width=0.2)
                # plt.xlim(0, 1)
                # plt.ylim(0, 1)

    plt.show()


plot_params(parameter_array[4:, :])
plot_params(parameter_array[:4, :])



nr, nc = parameter_array[4:, :].shape
u_train = ro.r.matrix(parameter_array[4:, :], nrow=nr, ncol=nc)

ro.r.assign("u_train2", u_train)
r('dim(u_train2)')


### P(x) -> 100 values at x-> 0:1 ###

pvec = (AllData['PVEC'])  # .newbyteorder('S')
# print(  np.unique( np.argwhere( np.isnan(pvec) )[:,0]) )

### silly hack for making sure the byteorder is R-readable
np.savetxt('pvec.txt', pvec)
pvec = np.loadtxt('pvec.txt')


######## Debugging interpolation issue ##########


### row 61 ( pvec value ) is extremely large ~ 1e10
### Either removing that or replacing it by correct value will fix the problem

## right now i'm deleting the 61st value (both pvec and corresponding parameter values
plt.figure(43)
plt.plot(pvec)

print pvec[61, :]
pvec = np.delete(pvec, (61), axis=0)
parameter_array = np.delete(parameter_array, (61), axis=0)


## right now I'm using a fake data for 61st entry
# pvec[61, :] = 0.5*(pvec[60, :] + pvec[62, :])


plt.figure(44)
plt.plot(pvec)



###########################################

nr, nc = pvec[4:, :].shape
y_train = ro.r.matrix(pvec[4:, :], nrow=nr, ncol=nc)

ro.r.assign("y_train2", y_train)
r('dim(y_train2)')



nr, nc = parameter_array[4:, :].shape
u_train = ro.r.matrix(parameter_array[4:, :], nrow=nr, ncol=nc)

ro.r.assign("u_train2", u_train)
r('dim(u_train2)')




########################### PCA ###################################
def PCA_decomp():
    Dicekriging = importr('DiceKriging')
    r('require(foreach)')
    # r('nrankmax <- 32')   ## Number of components
    ro.r.assign("nrankmax", nRankMax)

    r('svd(y_train2)')
    r('svd_decomp2 <- svd(y_train2)')
    r('svd_weights2 <- svd_decomp2$u[, 1:nrankmax] %*% diag(svd_decomp2$d[1:nrankmax])')

######################## GP FITTING ################################

## Build GP models
# This is evaluated only once for the file name. GP fitting is not required if the file exists.

def GP_fit():
    GPareto = importr('GPareto')

    ro.r('''
    
    GPmodel <- gsub("to", "",''' + GPmodel + ''')
    
    ''')

    r('''if(file.exists(GPmodel)){
            load(GPmodel)
        }else{
            models_svd2 <- list()
            for (i in 1: nrankmax){
                mod_s <- km(~., design = u_train2, response = svd_weights2[, i])
                models_svd2 <- c(models_svd2, list(mod_s))
            }
            save(models_svd2, file = GPmodel)
    
         }''')

    r('''''')


PCA_decomp()
GP_fit()

######################## GP PREDICTION ###############################


def GP_predict(para_array):
    ### Input: para_array -- 1D array [rho, sigma, tau, sspt]
    ### Output P(x) (size= 100)

    para_array = np.expand_dims(para_array, axis=0)

    nr, nc = para_array.shape
    Br = ro.r.matrix(para_array, nrow=nr, ncol=nc)

    ro.r.assign("Br", Br)

    r('wtestsvd2 <- predict_kms(models_svd2, newdata = Br , type = "UK")')
    r('reconst_s2 <- t(wtestsvd2$mean) %*% t(svd_decomp2$v[,1:nrankmax])')

    y_recon = np.array(r('reconst_s2'))

    return y_recon[0]


##################################### TESTING ##################################


plt.rc('text', usetex=True)   # Slower
plt.rc('font', size=12)  # 18 usually

plt.figure(999, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom=0.15)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$P(x)$')

ax1.axhline(y=1, ls='dotted')
# ax1.axhline(y=-1e-6, ls='dashed')
# ax1.axhline(y=1e-6, ls='dashed')

ax1.set_xlabel(r'$x$')

ax1.set_ylabel(r'emu/real - 1')
ax1.set_ylim(-1e-3, 1e-3)


# for x_id in [3, 23, 43, 64, 93, 109, 11]:
# for x_id in range(8,12):
for x_id in range(0, 4):

    time0 = time.time()
    x_decodedGPy = GP_predict(parameter_array[x_id])  ## input parameters
    time1 = time.time()
    print('Time per emulation %0.2f'% (time1 - time0), ' s')
    x_test = pvec[x_id]

    ax0.plot(x_decodedGPy, alpha=1.0, ls='--', label='emu')
    ax0.plot(x_test, alpha=0.9, label='real')
    plt.legend()

    ax1.plot(x_decodedGPy[1:] / x_test[1:] - 1)

plt.show()



######### TEMPLATE FOR MCMC LIKELIHOOD FUNCTION #######################
# For emcee

def lnlike(theta, x, y, yerr):
    p1, p2, p3, p4, p5 = theta

    new_params = np.array([p1, p2, p3, p4, p5])

    model = GP_predict(new_params)
    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model) / yerr) ** 2.))


