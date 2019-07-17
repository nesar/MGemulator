##### Generic packages ###############
import numpy as np
import matplotlib.pylab as plt
import time
import glob
import os

##### MCMC ###########################

import emcee
import pygtc

###### R kernel imports from rpy2 #####
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r
from rpy2.robjects.packages import importr


######### mcmc functions ###################


def lnprior(theta):
    p1, p2, p3, p4, p5, p6, p7 = theta
    # if 0.12 < p1 < 0.155 and 0.7 < p2 < 0.9:
    if para1[2] < p1 < para1[3] and para2[2] < p2 < para2[3] and para3[2] < p3 < para3[3] \
            and para4[2] < p4 < para4[3] and para5[2] < p5 < para5[3] and para6[2] < p6 < para6[
        3]and para7[2] < p7 < para7[3]:
        return 0.0
    return -np.inf


def lnlike_diag(theta, x, y, yerr):
    p1, p2, p3, p4, p5, p6, p7 = theta
    # new_params = np.array([p1, 0.0225, p2 , 0.74, 0.9])
    new_params = np.array([p1, p2, p3, p4, p5, p6, p7])

    model = GP_predict(new_params)
    mask = np.in1d(l, x)
    model_mask = model[mask]
    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model_mask) / yerr) ** 2.))


def lnlike(theta, x, y, icov):
    p1, p2, p3, p4, p5, p6, p7 = theta
    new_params = np.array([p1, p2, p3, p4, p5, p6, p7])

    model = GP_predict(new_params)
    mask = np.in1d(l, x)
    model_mask = model[mask]
    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    # return -0.5 * (np.sum(((y - model_mask) / yerr) ** 2.))

    data_vec_diff = y - model_mask
    loglike = -0.5*(data_vec_diff.T.dot(icov).dot(data_vec_diff))
    ### loglike = -0.5*(np.matmul(np.matmul(data_vec_diff.T, yerr), data_vec_diff))
    return  loglike




def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    # return lp + lnlike_diag(theta, x, y, yerr)
    return lp + lnlike(theta, x, y, yerr)


############################# PARAMETERS ##############################

dirIn = "deprecated_codes/cl_outputs/"  ## Input Cl files
paramIn = "lhc_128.txt"  ## 8 parameter file
nRankMax = 48 ## Number of basis vectors in truncated PCA
GPmodel = '"../RModels/R_GP_model_flat' + str(nRankMax) + '.RData"'  ## Double and single quotes are
# necessary

################################# I/O #################################
l = np.loadtxt(dirIn + 'xvals.txt')

RcppCNPy = importr('RcppCNPy')
# RcppCNPy.chooseCRANmirror(ind=1) # select the first mirror in the list


### 8 parameters:

# filelist = os.listdir(dirIn)
# filelist = glob.glob(dirIn + 'cls*')
filelist = glob.glob(dirIn + 'cls*')
filelist = sorted(filelist, key=lambda x: int(os.path.splitext(x)[0][72:]))

Px_flatflat = np.array([np.loadtxt(f) for f in filelist])

### Px_flatnan = np.unique(np.array(np.argwhere(np.isnan(Px_flatflat)) )[:,0])

Px_flatflat = Px_flatflat[: ,:, 1]


nan_idx = [~np.isnan(Px_flatflat).any(axis=1)]
Px_flatflat = Px_flatflat[nan_idx]
Px_flatlog = np.log10(Px_flatflat)


nr, nc = Px_flatlog.shape
y_train = ro.r.matrix(Px_flatlog, nrow=nr, ncol=nc)

ro.r.assign("y_train2", y_train)
r('dim(y_train2)')

parameter_array = np.loadtxt(paramIn)
parameter_array = parameter_array[nan_idx]

nr, nc = parameter_array.shape
u_train = ro.r.matrix(parameter_array, nrow=nr, ncol=nc)

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

    y_recon = 10**(y_recon)  ## GP fitting was done in log space, exponentiating here

    return y_recon[0]


##################################### TESTING ##################################


plt.rc('text', usetex=True)  # Slower
plt.rc('font', size=12)  # 18 usually

plt.figure(999, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom=0.15)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$P(x)$ (flat)')

ax1.axhline(y=1, ls='dotted')
# ax1.axhline(y=-1e-6, ls='dashed')
# ax1.axhline(y=1e-6, ls='dashed')

ax1.set_xlabel(r'$x$')

ax0.set_xscale('log')
ax1.set_xscale('log')
ax0.set_yscale('log')


ax1.set_ylabel(r'emu/real - 1')
ax1.set_ylim(-1e-5, 1e-5)

ax0.plot(l, (Px_flatflat.T), alpha=0.03, color='k')

for x_id in [3, 23, 43, 64, 83, 109]:
    time0 = time.time()
    x_decodedGPy = GP_predict(parameter_array[x_id])  ## input parameters
    time1 = time.time()
    print('Time per emulation %0.2f' % (time1 - time0), ' s')
    x_test = Px_flatflat[x_id]

    ax0.plot(l, x_decodedGPy, alpha=1.0, ls='--', label='emu')
    ax0.plot(l, (x_test), alpha=0.9, label='real')
    plt.legend()

    ax1.plot(x_decodedGPy[1:] / x_test[1:] - 1)





#################################################################################################
#################################################################################################
############################################## MCMC #############################################
#################################################################################################

#### parameters that define the MCMC

ndim = 7
nwalkers = 300  # 200 #600  # 500
nrun_burn = 50  # 50 # 50  # 300
nrun = 300  # 300  # 700
fileID = 1

########## REAL DATA with ERRORS #############################
# np.random.seed(42)

# Cl = np.loadtxt(filelist[0])[:,1]
# Cl = np.log(Cl + 0.1*Cl*np.random.standard_normal(Cl.shape[0]))
# emax = 0.05*Cl

########## REAL DATA with ERRORS #############################

dirDataIn = "./Data/"
Cl = np.loadtxt(dirDataIn + 'xip_vals.txt')
# Cl = np.log(Cl)
cov_mat = np.loadtxt(dirDataIn + 'cp_xip50.txt')

lsmax = 30
ls_cond = np.where(l < lsmax)


x = l
y = Cl
yerr_diag = np.sqrt(np.diag(cov_mat))



x = x[ls_cond]
y = y[ls_cond]
yerr_diag = yerr_diag[ls_cond]
# emax = emax[ls_cond][:,ls_cond][:,0,:]
cov_mat =  cov_mat[:len(ls_cond[0]), :len(ls_cond[0])]
## Only works if slicing is done at a corner.
# i.e., if ls_cond corresponds to continuous array entries in l
icov = np.linalg.inv(cov_mat)



# np.sqrt(yerr[::5])/Cl[::5]
ax0.errorbar(x[::], y[::], yerr= yerr_diag[::] , marker='o',
       color='k',
       ecolor='k',
       markerfacecolor='g',
       markersize = 2,
       capsize=0,
       linestyle='None')
# plt.show()

plt.savefig('Plots/PowerSpect_emu.pdf')


plt.figure(43)
plt.imshow(cov_mat)
plt.colorbar()
# plt.show()

plt.savefig('Plots/Cov_mat.pdf')

#### Cosmological Parameters ########################################


para1 = ["$\Omega_c h^2$", 0.1188, 0.12, 0.155]  # Actual 0.119
para2 = ["$\Omega_b h^2$", 0.02230, 0.0215, 0.0235]
para3 = ["$\sigma_8$", 0.8159, 0.7, 0.89]
para4 = ["$h$", 0.6774, 0.55, 0.85]
para5 = ["$n_s$", 0.9667, 0.85, 1.05]

para6 = ["$z_m$", 1.0, 0.5, 1.5] # z_m
para7 = ["FWHM", 0.25, 0.05, 0.5] # FWHM

#################### CHAIN INITIALIZATION ##########################

## 2 options

Uniform_init = True
if Uniform_init:
    # Choice 1: chain uniformly distributed in the range of the parameters
    pos_min = np.array( [para1[2], para2[2], para3[2], para4[2], para5[2], para6[2], para7[2]] )
    pos_max = np.array( [para1[3], para2[3], para3[3], para4[3], para5[3], para6[3], para7[3]] )
    psize = pos_max - pos_min
    pos0 = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]

True_init = False
if True_init:
    # Choice 2: chain is initialized in a tight ball around the expected values
    pos1 = [[para1[1] * 1.2, para2[1] * 0.8, para3[1] * 0.9, para4[1] * 1.1, para5[1] * 1.2,
             para6[1]*0.9, para6[1]*1.1] +
            1e-3 * np.random.randn(ndim) for i in range(nwalkers)]

MaxLikelihood_init = False
if MaxLikelihood_init:
    # Choice 2b: Find expected values from max likelihood and use that for chain initialization
    # Requires likehood function below to run first

    import scipy.optimize as op

    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, [para1[1], para2[1], para3[1], para4[1], para5[1], para6[1],
                               para7[1]],
                         args=(x, y, yerr))
    p1_ml, p2_ml, p3_ml, p4_ml, p5_ml, p6_ml, p7_ml = result["x"]
    print result['x']

    pos0 = [result['x'] + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

# Visualize the initialization

PriorPlot = False

if PriorPlot:
    fig = pygtc.plotGTC(pos0, labels=[para1[0], para2[0], para3[0], para4[0], para5[0], para6[0],
                                      para7[0]],
                        range=[[para1[2], para1[3]], [para2[2], para2[3]],
                               [para3[2], para3[3]],
                               [para4[2], para4[3]], [para5[2], para5[3]] , [para6[2], para6[3]]
                            , [para7[2], para7[3]] ],
                        truths=[para1[1], para2[1], para3[1], para4[1], para5[1], para6[1],
                                para7[1]])
    fig.set_size_inches(10, 10)

######### MCMC #######################


## Sample implementation :
# http://eso-python.github.io/ESOPythonTutorials/ESOPythonDemoDay8_MCMC_with_emcee.html
# https://users.obs.carnegiescience.edu/cburns/ipynbs/Emcee.html


# Let us setup the emcee Ensemble Sampler
# It is very simple: just one, self-explanatory line

# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr_diag))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, icov))

###### BURIN-IN #################

time0 = time.time()
# burnin phase
pos, prob, state = sampler.run_mcmc(pos0, nrun_burn)
sampler.reset()
time1 = time.time()
print('burn-in time:', time1 - time0)

###### MCMC ##################
time0 = time.time()
# perform MCMC
pos, prob, state = sampler.run_mcmc(pos, nrun)
time1 = time.time()
print('mcmc time:', time1 - time0)

samples = sampler.flatchain
samples.shape

###########################################################################
samples_plot = sampler.chain[:, :, :].reshape((-1, ndim))

np.savetxt('Data/Chains/SamplerPCA_mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
    nrun) + '.txt', sampler.chain[:, :, :].reshape((-1, ndim)))

####### FINAL PARAMETER ESTIMATES #######################################


samples_plot = np.loadtxt('Data/Chains/SamplerPCA_mcmc_ndim' + str(ndim) + '_nwalk' + str(
    nwalkers) + '_run' + str(nrun) + '.txt')

# samples = np.exp(samples)
p1_mcmc, p2_mcmc, p3_mcmc, p4_mcmc, p5_mcmc, p6_mcmc, p7_mcmc = map(lambda v: (v[1], v[2] - v[1],
                                                                               v[1] - v[0]) , zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print('mcmc results:', p1_mcmc[0], p2_mcmc[0], p3_mcmc[0], p4_mcmc[0], p5_mcmc[0], p6_mcmc[0],
      p7_mcmc[0])

####### CORNER PLOT ESTIMATES #######################################

CornerPlot = True
if CornerPlot:

    fig = pygtc.plotGTC(samples_plot,
                        paramNames=[para1[0], para2[0], para3[0], para4[0], para5[0], para6[0],
                                    para7[0]],
                        truths=[para1[1], para2[1], para3[1], para4[1], para5[1], para6[1],
                                para7[1]],
                        figureSize='MNRAS_page')  # , plotDensity = True, filledPlots = False,\smoothingKernel = 0, nContourLevels=3)

    fig.savefig('Plots/pygtcPCA_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun) +  '.pdf')

####### FINAL PARAMETER ESTIMATES #######################################
#
# plt.figure(1432)
#
# xl = np.array([0, 10])
# for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
#     plt.plot(xl, m*xl+b, color="k", alpha=0.1)
# plt.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
# plt.errorbar(x, y, yerr=yerr, fmt=".k", alpha=0.1)



####### SAMPLER CONVERGENCE #######################################

ConvergePlot = False
if ConvergePlot:
    fig = plt.figure(13214)
    plt.xlabel('steps')
    ax1 = fig.add_subplot(7, 1, 1)
    ax2 = fig.add_subplot(7, 1, 2)
    ax3 = fig.add_subplot(7, 1, 3)
    ax4 = fig.add_subplot(7, 1, 4)
    ax5 = fig.add_subplot(7, 1, 5)
    ax6 = fig.add_subplot(7, 1, 6)
    ax7 = fig.add_subplot(7, 1, 7)

    ax1.plot(np.arange(nrun), sampler.chain[:, :, 0].T, lw=0.2, alpha=0.9)
    ax1.text(0.9, 0.9, para1[0], horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes, fontsize=20)
    ax2.plot(np.arange(nrun), sampler.chain[:, :, 1].T, lw=0.2, alpha=0.9)
    ax2.text(0.9, 0.9, para2[0], horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes, fontsize=20)
    ax3.plot(np.arange(nrun), sampler.chain[:, :, 2].T, lw=0.2, alpha=0.9)
    ax3.text(0.9, 0.9, para3[0], horizontalalignment='center', verticalalignment='center',
             transform=ax3.transAxes, fontsize=20)
    ax4.plot(np.arange(nrun), sampler.chain[:, :, 3].T, lw=0.2, alpha=0.9)
    ax4.text(0.9, 0.9, para4[0], horizontalalignment='center', verticalalignment='center',
             transform=ax4.transAxes, fontsize=20)
    ax5.plot(np.arange(nrun), sampler.chain[:, :, 4].T, lw=0.2, alpha=0.9)
    ax5.text(0.9, 0.9, para5[0], horizontalalignment='center', verticalalignment='center',
             transform=ax5.transAxes, fontsize=20)
    ax6.plot(np.arange(nrun), sampler.chain[:, :, 5].T, lw=0.2, alpha=0.9)
    ax6.text(0.9, 0.9, para6[0], horizontalalignment='center', verticalalignment='center',
             transform=ax6.transAxes, fontsize=20)
    ax7.plot(np.arange(nrun), sampler.chain[:, :, 5].T, lw=0.2, alpha=0.9)
    ax7.text(0.9, 0.9, para7[0], horizontalalignment='center', verticalalignment='center',
             transform=ax7.transAxes, fontsize=20)
    plt.show()

    fig.savefig('Plots/convergencePCA_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun) + '.pdf')



### FUTURE


# Use Mira-Titan's values for P(x)
# Get CosmoDC2's P(x) values
# Currently emulating over just 7 parameters, not 8
# rerun for diff covariance matrix with same xvals.txt
