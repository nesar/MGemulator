import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from itertools import cycle
import matplotlib.ticker as ticker
from matplotlib import gridspec


import numpy as np  
import time
import pickle
import os
from sklearn.decomposition import PCA
import gpflow

############################# PARAMETERS ##############################

dataDir = "./Data/" ## Data folder
modelDir = "./Models/" ## Data folder
plotsDir = "./Plots/allPlots/" ## Data folder

nRankMax = [2, 4, 8, 12, 16, 32][4]  ## Number of basis vectors in truncated PCA
## Increasing nRankMax will increase emulation precision (asymptotically), but reduce the speed

del_idx = [5, 25, 4, 42]  ## Random holdouts (not used in training, reserved for validation) 

snap_ID_arr = np.arange(100)

for snap_ID in snap_ID_arr:


    ############################# PARAMETERS ##############################

    dataDir = "./Data/Emulator213bins/" ## Data folder
    fileIn = dataDir + 'ratiosbins_' + str(snap_ID) + '.txt'



    paramIn = dataDir + 'mg.design'


    az = np.loadtxt(dataDir + 'timestepsCOLA.txt', skiprows=1) 
    fileIn = dataDir + 'ratiosbins_' + str(snap_ID) + '.txt'

    GPmodel = modelDir + 'Matern1noARDGP_smooth_rank' + str(nRankMax) + 'snap' + str(snap_ID)  ## Double and single quotes are necessary
    PCAmodel = modelDir + 'Matern1noARDPCA_smooth_rank' + str(nRankMax) + 'snap' + str(snap_ID)  ## Double and single quotes are necessary

    print(GPmodel)
    ################################# I/O #################################
    loadFile = np.loadtxt(fileIn)
    PmPl_all = loadFile[:, 1:].T
    kvals = loadFile[:,0]


    parameter_array_all = np.loadtxt(paramIn)


    ############## rescaling ##############


    def rescale01(f):
        return np.min(f), np.max(f), (f - np.min(f)) / (np.max(f) - np.min(f))


    def scale01(fmin, fmax, f):
        return (f - fmin) / (fmax - fmin)
    #     return f*(fmax - fmin) + fmin


    lhd = np.zeros_like(parameter_array_all)
    lhdmin = np.zeros_like(parameter_array_all[1])
    lhdmax = np.zeros_like(parameter_array_all[1])

    for i in range(parameter_array_all.shape[1]):
        lhdmin[i], lhdmax[i], lhd[:, i] = rescale01(parameter_array_all[:, i])
    

    parameter_array_all = lhd
    np.savetxt(dataDir+'paralims.txt', np.array([lhdmin, lhdmax]))


    ############## rescaling ##############


    ## Removing hold-out test points
    parameter_array = np.delete(parameter_array_all, del_idx, axis=0)
    PmPl = np.delete(PmPl_all, del_idx, axis=0)

    np.savetxt(dataDir+'kvals.txt', kvals)

    lhdmin, lhdmax = np.loadtxt(dataDir + 'paralims.txt')

    #### adding smoothing filter ########

    import scipy.signal
    yhat = scipy.signal.savgol_filter(PmPl[:,:], 51, 3) # window size 51, polynomial order 3
    y_train = yhat


    ############################# Plot the input parameter distribution ##############################

    allLabels = [r'${\Omega}_m$', r'$n_s$', r'${\sigma}_8$', r'$f_{R_0}$', r'$n$']


    lhd = np.zeros_like(parameter_array_all)
    for i in range(parameter_array_all.shape[1]):
        _,_,lhd[:, i] = rescale01(parameter_array_all[:, i])
        
    def plot_params(lhd):
        f, a = plt.subplots(lhd.shape[1], lhd.shape[1], sharex=True, sharey=True, figsize=(10, 10) )
        plt.suptitle('lhc design (rescaled parameters)', fontsize = 28)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.rcParams.update({'font.size': 8})

        for i in range(lhd.shape[1]):
            for j in range(i + 1):
                # print(i, j)
                if (i != j):
                    a[i, j].scatter(lhd[:, i], lhd[:, j], s=5)
                    a[i, j].grid(True)
                else:
                    hist, bin_edges = np.histogram(lhd[:, i], density=True, bins=64)
                    a[i, i].text(0.4, 0.4, allLabels[i], size = 'xx-large')

                    a[i, i].bar(bin_edges[:-1], hist / hist.max(), width=0.2, alpha = 0.1)


        # plt.show()


    # plot_params(lhd)

    ########################### PCA ###################################
    # set up pca compression


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


    ######################## GP FITTING ################################

    ## Build GP models
    # This is evaluated only once for the file name. GP fitting is not required if the file exists.
    

    def GPflow_fit(parameter_array, weights, fname= GPmodel):
        # kern = gpflow.kernels.Matern52(input_dim = np.shape(parameter_array)[1]) #, ARD=True)
        kern = gpflow.kernels.Matern12(input_dim = np.shape(parameter_array)[1], ARD=False)
    #     m1 = GPy.models.GPRegression(parameter_array, weights, kernel=kern)
        m = gpflow.models.GPR(parameter_array, weights, kern=kern, mean_function=None)
    #     print_summary(m)
        m.likelihood.variance.assign(0.01)
    #     m.kern.lengthscales.assign([0.3, 0.1, 0.2, 0.3, 0.1])
        # m.kern.lengthscales.assign([25, 65, 15 ,1, 1])
        
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m)
        m.as_pandas_table()
        
        from pathlib import Path

        print(f'GPR lengthscales =', m.kern.lengthscales.value)

        
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

    # pca_model, pca_weights, pca_bases = PCA_compress(PmPl, nComp=nRankMax)
    pca_model, pca_weights, pca_bases = PCA_compress(y_train, nComp=nRankMax)

    print('----------------')
    print(parameter_array.shape)
    print(pca_weights.shape)
    print('----------------')

    GPflow_fit(parameter_array, pca_weights)


    ctx_for_loading = gpflow.saver.SaverContext(autocompile=False)
    saver = gpflow.saver.Saver()

    m1 = saver.load(GPmodel, context=ctx_for_loading)
    m1.clear()
    m1.compile()

    pca_model = pickle.load(open(PCAmodel, 'rb'))

    plt.rc('text', usetex=True)  # Slower
    plt.rc('font', size=18)  # 18 usually

    plt.figure(999 + snap_ID, figsize=(14, 12))
    from matplotlib import gridspec

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(hspace=0.02, left=0.2, bottom=0.15)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.set_ylabel(r'$p(x)$', fontsize=25)
    ax1.set_xlabel(r'$x$', fontsize=25)
    ax1.set_ylabel(r'$p_{emu}/p_{num} - 1$', fontsize = 18)
    # ax1.set_ylim(-5e-2, 5e-2)

    ax0.set_xscale('log')
    # ax0.set_yscale('log')
    ax1.set_xscale('log')

    ax1.axhline(y=0, ls='dashed')

    color_id = 0
    for x_id in del_idx:
        color_id = color_id + 1
        time0 = time.time()
    #     x_decoded_new = Emu(parameter_array_all[x_id], PCAmodel='PCA_model', GPmodel='GPy_model')
        x_decoded_new = Emu(parameter_array_all[x_id])
        x_decoded_smooth = scipy.signal.savgol_filter(x_decoded_new , 51, 6)

        time1 = time.time()
        print('Time per emulation %0.5f' % (time1 - time0), ' s')
        ax0.plot(kvals, x_decoded_new, alpha=1.0, lw = 1.5, ls='--', label='emu', dashes=(10, 10), color=plt.cm.Set1(color_id))
        ax0.plot(kvals, x_decoded_smooth, alpha=1.0, lw = 1.5, ls='--', label='emu', dashes=(10, 10), color=plt.cm.Set1(color_id))

    #     x_test = PmPl_all[x_id]
        x_test = scipy.signal.savgol_filter(PmPl_all[x_id], 51, 6)

        ax0.plot(kvals, x_test, alpha=0.4, label='real', color=plt.cm.Set1(color_id))

        ax1.plot(kvals, (x_decoded_smooth / (x_test) ) - 1, ls='--', dashes=(10, 2), color=plt.cm.Set1(color_id))


    ax0.set_xticklabels([])
    plt.savefig(plotsDir + 'Matern1noARDPemu_rank'  + str(snap_ID) + '.png', figsize=(28, 24), bbox_inches="tight")


    pca_model = pickle.load(open(PCAmodel , 'rb'))


    ## Calling the Emulator function with 5 arguements [Om, ns, sigma8, fR0, n]

    print('sample emulated value:', Emu(np.array([1, 1, 1, 1, 1]) ) )
    print(50*'=')

    #########################################################

    colorList = ['r', 'g', 'k', 'b', 'brown', 'orange', 'purple', 'darkslateblue', 'darkkhaki']

    plt.rc('text', usetex=True)  # Slower
    plt.rc('font', size=18)  # 18 usually


    plt.figure(999 + snap_ID, figsize=(14, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(hspace=0.1, left=0.2, bottom=0.15, wspace=0.25)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.set_ylabel(r'$P_{MG}(k)/P_{{\Lambda}CDM}(k)$')
    ax1.set_xlabel(r'$k$[h/Mpc]')
    ax1.axhline(y=0, ls='dashed')

    # ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax1.set_xscale('log')

    ax1.set_ylabel(r'emu/test - 1')

    ax0.plot(kvals, PmPl_all.T, alpha=0.15, color='k')


    ax0.set_xlim(kvals[0], kvals[-1])
    ax1.set_xlim(kvals[0], kvals[-1])
    ax1.set_ylim(-9e-2, 9e-2)

    # del_idx = [0, 1, 2, 5, 6, 7, 8]
    color_id = 0
    for x_id in del_idx:
        print(x_id)
        color_id = color_id + 1
        time0 = time.time()
        x_decodedGPy = Emu(parameter_array_all[x_id])  ## input parameters
        time1 = time.time()
        print('Time per emulation %0.4f' % (time1 - time0), ' s')
        x_test = PmPl_all[x_id]

        ax0.plot(kvals, x_decodedGPy, alpha=1.0, ls='--', lw = 1.9, dashes=(5, 5), label='emu', color=colorList[color_id])
        ax0.plot(kvals, x_test, alpha=0.7, label='test', color=colorList[color_id])
        ax1.plot( kvals, (x_decodedGPy[:]) / (x_test[:])  - 1, color=colorList[color_id])
        
    start, end = ax0.get_ylim()
    ax0.yaxis.set_ticks((np.arange(start, end, 0.1)))
    ax0.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))


    ax0.set_xticklabels([])
    plt.savefig(plotsDir + 'Matern1noARDMGemu_rank' + str(snap_ID) + '.png', figsize=(28, 24), bbox_inches="tight")
    plt.show()

    allMax = np.max(parameter_array, axis = 0)
    allMin = np.min(parameter_array, axis = 0)
    allMean = np.mean(parameter_array, axis = 0)


    print(allMin)
    print(allMax)
    print(allMean)
    Pk_mean = Emu(allMean) 

    PlotCls = True

    if PlotCls:

        numPlots = 5
        fig, ax = plt.subplots(5,2, figsize = (15,26))
        plt.subplots_adjust(wspace=0.25)

        for paramNo in range(5):
            print(paramNo)
            para_range = np.linspace(allMin[paramNo], allMax[paramNo], numPlots)

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
        fig.savefig(plotsDir + "Matern1noARDsensitivity_snap" + str(snap_ID) + ".png",  bbox_inches="tight", dpi=200)


    # plt.show()
    plt.clf()
    plt.close('all')
