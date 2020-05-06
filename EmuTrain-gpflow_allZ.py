"""
GP emulation
"""
import matplotlib as mpl
mpl.use('Agg')

##### Packages ###############
import numpy as np  
import matplotlib.pylab as plt
import time
import pickle
import os
from sklearn.decomposition import PCA
import matplotlib.ticker as ticker
import gpflow
import scipy.signal
from pathlib import Path
from itertools import cycle
from matplotlib import gridspec

def rescale01(f):
    return np.min(f), np.max(f), (f - np.min(f)) / (np.max(f) - np.min(f))

def scale01(fmin, fmax, f):
    return (f - fmin) / (fmax - fmin) ## This is not an unscaling operation

def SmoothPk(Pk_in):
    Pk_out = scipy.signal.savgol_filter(Pk_in, 51, 3)
    return Pk_out  

########################### PCA ###################################

def PCA_compress(x, nComp):
    # x is in shape (nparams, nbins)
    pca_model = PCA(n_components=nComp)
    principalComponents = pca_model.fit_transform(x)
    pca_bases = pca_model.components_

    print("original shape:   ", x.shape)
    print("transformed shape:", principalComponents.shape)
    print("bases shape:", pca_bases.shape)

    import pickle
    pickle.dump(pca_model, open(PCAmodel, 'wb'))

    return pca_model, np.array(principalComponents), np.array(pca_bases)


def GPflow_fit(parameter_array, weights, fname):
    kern1 = gpflow.kernels.RBF(input_dim = np.shape(parameter_array)[1], ARD=True)
    kern2 = gpflow.kernels.RBF(input_dim = np.shape(parameter_array)[1], ARD=False)

    kern = kern1 + kern2 
    # m1 = GPy.models.GPRegression(parameter_array, weights, kernel=kern)
    m = gpflow.models.GPR(parameter_array, weights, kern=kern, mean_function=None)
    # print_summary(m)
    m.likelihood.variance.assign(1e-10)
    # m.kern.lengthscales.assign([25, 65, 15 ,1, 1])
    # m.kern.lengthscales.assign(0.5)
        
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    m.as_pandas_table()
    # print(f'GPR lengthscales =', m.kern.lengthscales.value)
 
    path = Path(GPmodel)
    if path.exists():
        path.unlink()
        
    saver = gpflow.saver.Saver()
    saver.save(fname, m)

######################## GP PREDICTION FUNCTIONS ###############################

def GPy_predict(para_array):
    m1p = m1.predict_f(para_array)  # [0] is the mean and [1] the predictive
    W_predArray = m1p[0]
    W_varArray = m1p[1]
    return W_predArray, W_varArray


def Emu(para_array):
    if len(para_array.shape) == 1:
        W_predArray, _ = GPy_predict(np.expand_dims(para_array, axis=0))
        x_decoded = pca_model.inverse_transform(W_predArray)
        return np.squeeze(x_decoded)#[0]

def EmuPlusMinus(para_array):
    if len(para_array.shape) == 1:
        W_predArray, W_varArray = GPy_predict(np.expand_dims(para_array, axis=0))
        x_decoded = pca_model.inverse_transform(W_predArray)
        x_decoded_plus = pca_model.inverse_transform(W_predArray + np.sqrt(W_varArray))
        x_decoded_minus = pca_model.inverse_transform(W_predArray - np.sqrt(W_varArray))
        return np.squeeze(x_decoded), np.squeeze(x_decoded_plus), np.squeeze(x_decoded_minus)



############################# PARAMETERS ##############################

dataDir = "./Data/Fixedn_val_latest/" ## Data folder
modelDir = "./Models/Fixedn_val_latest/" ## Data folder
plotsDir = "./Plots/Fixedn_val_latest/" ## Data folder


nRankMax = [2, 4, 5, 6, 7, 8, 12, 16, 32][3]  ## Number of basis vectors in truncated PCA
del_idx = [50, 51, 52, 53, 54]# [8, 12, 3, 43] ## Random holdouts (not used in training, reserved for validation) 

snap_ID_arr = np.arange(100)
for snap_ID in snap_ID_arr: 
    ############################# PARAMETERS ##############################
    fileIn = dataDir + 'ratiosbinsnew_' + str(snap_ID) + '.txt'
    paramIn = dataDir + 'mg_log_val_2.design'
    az = np.loadtxt(dataDir + 'timestepsCOLA.txt', skiprows=1) 
    fileIn = dataDir + 'ratiosbinsnew_' + str(snap_ID) + '.txt'
    GPmodel = modelDir + 'nCorrLogfixedGP_smooth_rank' + str(nRankMax) + 'snap' + str(snap_ID)  
    PCAmodel = modelDir + 'nCorrLogfixedPCA_smooth_rank' + str(nRankMax) + 'snap' + str(snap_ID) 
    print(GPmodel)
    ################################# I/O ########
 
    print('Training snapshot: ', str(snap_ID))  
    print(40*'=*')

    loadFile = np.loadtxt(fileIn)
    PmPl_all = loadFile[:, 1:].T
    kvals = loadFile[:,0]

    parameter_array_all = np.loadtxt(paramIn)
    parameter_array_all[:, 3] = np.log10(parameter_array_all[:, 3])                
    
    #### adding smoothing filter ########
    yhat = SmoothPk(PmPl_all[:,:]) # window size 51, polynomial order 3        
    ############## rescaling ##############

    lhd = np.zeros_like(parameter_array_all)
    lhdmin = np.zeros_like(parameter_array_all[1])
    lhdmax = np.zeros_like(parameter_array_all[1])

    for i in range(parameter_array_all.shape[1]):
        lhdmin[i], lhdmax[i], lhd[:, i] = rescale01(parameter_array_all[:, i])
    
    parameter_array_log_scaled = lhd
    np.savetxt(dataDir+'paralims_nCorr_val_2.txt', np.array([lhdmin, lhdmax]))

    ############## rescaling ##############

    ## Removing hold-out test points
    parameter_array = np.delete(parameter_array_log_scaled, del_idx, axis=0)
    # PmPl = np.delete(PmPl_all, del_idx, axis=0)
    y_train = np.delete(yhat, del_idx, axis=0)

    # Om=Om_f, ns=ns_f, s8=s8_f, fR0=fr0_f4, n=n_f, z=z_f   
    print(lhdmin)
    print(lhdmax)    
    lhdmin, lhdmax = np.loadtxt(dataDir + 'paralims_nCorr_val_2.txt')   

    allLabels = [r'${\Omega}_m$', r'$n_s$', r'${\sigma}_8$', r'$log(f_{R_0})$', r'$n$']


    lhd = np.zeros_like(parameter_array)
    for i in range(parameter_array.shape[1]):
        _,_,lhd[:, i] = rescale01(parameter_array[:, i]) 


    ############### PCA + GP fitting ##############
    pca_model, pca_weights, pca_bases = PCA_compress(y_train, nComp=nRankMax)

    print('----------------')
    print(parameter_array.shape)
    print(pca_weights.shape)
    print('----------------')

    GPflow_fit(parameter_array, pca_weights, GPmodel)

    ########## Testing ##############

    ctx_for_loading = gpflow.saver.SaverContext(autocompile=False)
    saver = gpflow.saver.Saver()

    m1 = saver.load(GPmodel, context=ctx_for_loading)
    m1.clear()
    m1.compile()

    pca_model = pickle.load(open(PCAmodel, 'rb'))

    ##################################### TESTING ##################################

    plt.rc('font', size=18)  # 

    PlotPrior = True

    if PlotPrior:

        plt.figure(999, figsize=(14, 12))

        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        gs.update(hspace=0.1, left=0.2, bottom=0.15, wspace=0.25)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        ax0.set_ylabel(r'$P_{MG}(k)/P_{{\Lambda}CDM}(k)$',  fontsize = 18)

        ax1.set_xlabel(r'$k$[h/Mpc]',  fontsize = 18)
        ax1.axhline(y=0, ls='dashed')


        ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax1.set_ylabel(r'emu/test - 1',  fontsize = 18)
        # ax1.set_ylim(-5e-2, 5e-2)

        ax0.plot(kvals, PmPl_all.T, alpha=0.15, color='k')

        start, end = ax0.get_ylim()
        ax0.yaxis.set_ticks((np.arange(start, end, 0.1)))
        ax0.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))


        ax0.set_xlim(kvals[0], kvals[-1])
        ax1.set_xlim(kvals[0], kvals[-1])
        # ax0.set_xlim(kvals[0], 1.0)
        # ax1.set_xlim(kvals[0], 1.0)
        ax0.set_xticklabels([])


        color_id = 0
        for x_id in del_idx:
            color_id = color_id + 1

            time0 = time.time()
            x_decodedGPy = Emu(parameter_array_log_scaled[x_id])  ## input parameters
            time1 = time.time()
            print('Time per emulation %0.3f' % (time1 - time0), ' s')
            x_test = PmPl_all[x_id]
            x_test_smooth = yhat[x_id]
            # x_test_smooth2 = SmoothPk(x_test)

            ax0.plot(kvals, x_decodedGPy, alpha=1.0, ls='--', lw = 1.9, dashes=(5, 5), label='emu', color=plt.cm.Set1(color_id))
            ax0.plot(kvals, x_test, alpha=0.7, ls='-.', label='test '+str(x_id), color=plt.cm.Set1(color_id))
            ax0.plot(kvals, x_test_smooth, alpha=0.7, ls='-', label='smooth '+str(x_id), color=plt.cm.Set1(color_id))
            # ax0.plot(kvals, x_test_smooth2, alpha=0.7, ls='--', label='smooth2 '+str(x_id), color=plt.cm.Set1(color_id))

            # ax1.plot( kvals, (x_decodedGPy[:]) / (x_test[:])  - 1, color=plt.cm.Set1(color_id))
            ax1.plot( kvals, (x_decodedGPy[:]) / (x_test_smooth)  - 1, color=plt.cm.Set1(color_id))

    # ax0.text(0.07, 1.4, 'z = %0.2f'%z_ID, fontsize= 18, style='italic')

    ax0.legend()
    plt.savefig(plotsDir + "Emu_snap" + str(snap_ID) + ".png",  bbox_inches="tight", dpi=200)

    plt.clf()

    ##################################### SENSITIVITY ##################################



    allMax = np.max(parameter_array, axis = 0)
    allMin = np.min(parameter_array, axis = 0)
    allMean = np.mean(parameter_array, axis = 0)

    #allMean = 0.5*(allMax - allMin)

    print(allMin)
    print(allMax)
    print(allMean)
    Pk_mean = Emu(allMean) 


    PlotCls = True

    numPlots = 5

    fig, ax = plt.subplots(5,2, figsize = (15,26))
    plt.subplots_adjust(wspace=0.25)

    if PlotCls:
        for paramNo in range(5):
            print(paramNo)
            para_range = np.linspace(1.05*allMin[paramNo], 0.95*allMax[paramNo], numPlots)

            #plt.figure(32)
            lines = ["-","-.","--",":"]
            linecycler = cycle(lines)
            dashList = [(6,2),(10,1),(5,5),(3,3,2,2),(5,2,20,2)]
            colorList = ['r', 'g', 'k', 'b', 'brown']


            for plotID in range(numPlots):
                para_plot = np.copy(allMean)
                para_plot[paramNo] = para_range[plotID]  #### allMean gets changed everytime!!
                x_decodedGPy = Emu(para_plot) 
                lineObj = ax[4-paramNo,0].plot(kvals, x_decodedGPy, lw= 1.5, linestyle='--', dashes=dashList[plotID], color = colorList[plotID], label = allLabels[paramNo] + ' = %.1e'%para_range[plotID])
                #ax[paramNo,0].set_ylim(9.9, None)
                #ax[4-paramNo,0].set_yscale('log')
                ax[4-paramNo,0].set_xscale('log')
                ax[4-paramNo,0].set_ylabel(r'$P_{MG}(k)/P_{{\Lambda}CDM}(k)$')
                ax[4-paramNo,0].set_xlabel('$k$[h/Mpc]')
                #ticks = np.linspace(np.min(10**x_decodedGPy), np.max(10**x_decodedGPy), 5)
                #ticks = np.array([10, 15, 20, 25, 30, 35, 40])
                #ax[4-paramNo,0].set_yticks(ticks, minor = True)
                ax[4-paramNo,0].set_yticks([], minor = True)
                ax[4-paramNo,0].legend(iter(lineObj), para_range.round(decimals=2), title = allLabels[paramNo])
                ax[4-paramNo,0].legend()
                #ax[paramNo,0].legend(title = allLabels[paramNo])
                #ax[paramNo,1].set_yscale('log')
                ax[4-paramNo,1].set_xscale('log')
                ax[4-paramNo,1].set_ylabel(r'$\Delta f / f_0$')
                ax[4-paramNo,1].set_xlabel('$k$[h/Mpc]')
                #ax[paramNo,0].legend(iter(lineObj), para_range.round(decimals=2), title = allLabels[paramNo])
                #ax[paramNo,0].legend(title = allLabels[paramNo])
                ax[4-paramNo,1].plot(kvals, (x_decodedGPy)/(Pk_mean) - 1, lw= 1.5, linestyle='--', dashes=dashList[plotID], color = colorList[plotID], label = para_range[plotID] )


            start, end = ax[4-paramNo, 0].get_ylim()
            ax[4-paramNo, 0].yaxis.set_ticks( (np.arange(start, end, 0.1)))
            ax[4-paramNo, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))


    fig.savefig(plotsDir + "sensitivity_snap" + str(snap_ID) + ".png",  bbox_inches="tight", dpi=200)

    plt.clf()
    plt.close('all')







# par = [0.14, 1.0, 0.7, -5, 1.0] #params[11]
# # par[3] = np.log10(par[3])
# scaled_par = scale01(lhdmin, lhdmax, par)
# print(par)
# print( (scaled_par))

# # x_id = 46
# # 
# # 

# # print( params[np.argmin(params[:,3])])
# # print(parameter_array_all[x_id])
# plt.figure(12)
# Emu(scaled_par)
# plt.plot(kvals, Emu(scaled_par), '--')
# plt.plot(kvals, EmuPlusMinus(scaled_par)[0])
# plt.fill_between(kvals, EmuPlusMinus(scaled_par)[1], EmuPlusMinus(scaled_par)[2], alpha = 0.5)
# plt.plot(kvals, np.ones_like(kvals), 'k--')

# plt.ylabel(r'$p_{MG}(k)/p_{LCDM}(k)$')
# plt.xlabel(r'$k/h$')

# plt.show()

