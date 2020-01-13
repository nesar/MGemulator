#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
GP emulation -- bokeh viz

bokeh serve --show EmuTrain_bokeh.py --port 8000 --allow-websocket-origin=*

"""
##### Packages ###############
import numpy as np  
# import matplotlib.pylab as plt
import time
import pickle
import os
from sklearn.decomposition import PCA
import GPy
import matplotlib.ticker as ticker


# In[2]:


############################# PARAMETERS ##############################

dataDir = "./Data/" ## Data folder
modelDir = "./Models/" ## Data folder
plotsDir = "./Plots/" ## Data folder

nRankMax = [2, 4, 8, 12, 16, 32][0]  ## Number of basis vectors in truncated PCA
## Increasing nRankMax will increase emulation precision (asymptotically), but reduce the speed

del_idx = [5, 25, 15]  ## Random holdouts (not used in training, reserved for validation) 
snap_ID = 97


############################# PARAMETERS ##############################

dataDir = "./Data/Emulator213bins/" ## Data folder
fileIn = dataDir + 'ratiosbins_' + str(snap_ID) + '.txt'



paramIn = dataDir + 'mg.design'
fileIn = dataDir + ['ratios.txt', 'PMG.txt'][0]



# plotsDir = "./Plots/" ## Data folder
# dataDir = "./Data/Emulator_data/" ## Data folder
# dataDir = "./Data/Emulator213bins/" ## Data folder

# paramIn = dataDir + 'mg.design'  ## parameter file



az = np.loadtxt(dataDir + 'timestepsCOLA.txt', skiprows=1) 
fileIn = dataDir + 'ratiosbins_' + str(snap_ID) + '.txt'
GPmodel = modelDir + 'GPy_model_213Smooth_rank' + str(nRankMax) + 'snap' + str(snap_ID) 
# print(GPmodel)
################################# I/O #################################


# In[3]:


loadFile = np.loadtxt(fileIn)
PmPl_all = loadFile[:, 1:].T
kvals = loadFile[:,0]


parameter_array_all = np.loadtxt(paramIn)


############## rescaling ##############


def rescale01(f):
    return (f - np.min(f)) / (np.max(f) - np.min(f))


lhd = np.zeros_like(parameter_array_all)
for i in range(parameter_array_all.shape[1]):
    lhd[:, i] = rescale01(parameter_array_all[:, i])
   

parameter_array_all = lhd

PmPl_all = rescale01(loadFile[:, 1:].T)

############## rescaling ##############


## Removing hold-out test points
parameter_array = np.delete(parameter_array_all, del_idx, axis=0)
PmPl = np.delete(PmPl_all, del_idx, axis=0)




############################# Plot the input parameter distribution ##############################

allLabels = [r'${\Omega}_m$', r'$n_s$', r'${\sigma}_8$', r'$f_{R_0}$', r'$n$']

def rescale01(f):
    return (f - np.min(f)) / (np.max(f) - np.min(f))

lhd = np.zeros_like(parameter_array_all)
for i in range(parameter_array_all.shape[1]):
    lhd[:, i] = rescale01(parameter_array_all[:, i])


########################### PCA ###################################
# set up pca compression
from sklearn.decomposition import PCA


def PCA_compress(x, nComp):
    # x is in shape (nCosmology, nbins)
    pca_model = PCA(n_components=nComp)
    principalComponents = pca_model.fit_transform(x)
    pca_bases = pca_model.components_

#     print("original shape:   ", x.shape)
#     print("transformed shape:", principalComponents.shape)
#     print("bases shape:", pca_bases.shape)

    import pickle
    pickle.dump(pca_model, open(modelDir + 'GPy_PCA_model' + str(nRankMax), 'wb'))

    return pca_model, np.array(principalComponents), np.array(pca_bases)


######################## GP FITTING ################################

## Build GP models
# This is evaluated only once for the file name. GP fitting is not required if the file exists.

def GPy_fit0(parameter_array, weights, fname= GPmodel):
    kern = GPy.kern.Matern52( np.shape(parameter_array)[1], 0.1)
    m1 = GPy.models.GPRegression(parameter_array, weights, kernel=kern)
#     m1 = GPy.models.GPRegression(parameter_array, weights)

#     m1.Gaussian_noise.variance.constrain_fixed(1e-12)
#     m1.optimize(messages=True)
    m1.save_model(fname + str(nRankMax), compress=True, save_data=True)
    

def GPy_fit(parameter_array, weights, fname= GPmodel):
#     kern = GPy.kern.RBF(input_dim= np.shape(parameter_array)[1], lengthscale=1000, variance=1e-10)
    kern = GPy.kern.Matern52(input_dim= np.shape(parameter_array)[1])
    m1 = GPy.models.GPRegression(parameter_array, weights, kernel=kern)
    m1.randomize()
    m1.optimize(optimizer = 'lbfgs', messages=True)
    m1.save_model(fname, compress=True, save_data=True)
       


# In[7]:


######################## GP PREDICTION FUNCTIONS ###############################

def GPy_predict(para_array):
    m1p = m1.predict(para_array)  # [0] is the mean and [1] the predictive
    W_predArray = m1p[0]
    W_varArray = m1p[1]
    return W_predArray, W_varArray


def Emu(para_array):
    if len(para_array.shape) == 1:
        W_predArray, _ = GPy_predict(np.expand_dims(para_array, axis=0))
        x_decoded = pca_model.inverse_transform(W_predArray)
        return x_decoded[0]

    else:
        W_predArray, _ = GPy_predict(para_array)
        x_decoded = pca_model.inverse_transform(W_predArray)
        return x_decoded.T


# In[8]:


pca_model, pca_weights, pca_bases = PCA_compress(PmPl, nComp=nRankMax)
# GPy_fit(parameter_array, pca_weights)



# m1 = GPy.models.GPRegression.load_model(GPmodel)

# pca_model = pickle.load(open(modelDir + 'GPy_PCA_model' + str(nRankMax), 'rb'))

m1 = GPy.models.GPRegression.load_model(GPmodel + '.zip')

pca_model = pickle.load(open(modelDir + 'GPy_PCA_model' + str(nRankMax), 'rb'))


## Calling the Emulator function with 5 arguements [Om, ns, sigma8, fR0, n]

# print('sample emulated value:', Emu(np.array([1, 1, 1, 1, 1]) ) )
# print(50*'=')

#########################################################

# colorList = ['r', 'g', 'k', 'b', 'brown', 'orange', 'purple', 'darkslateblue', 'darkkhaki']

# plt.rc('text', usetex=True)  # Slower
# plt.rc('font', size=18)  # 18 usually


# plt.figure(999, figsize=(14, 12))
# from matplotlib import gridspec

# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
# gs.update(hspace=0.1, left=0.2, bottom=0.15, wspace=0.25)
# ax0 = plt.subplot(gs[0])
# ax1 = plt.subplot(gs[1])

# ax0.set_ylabel(r'$P_{MG}(k)/P_{{\Lambda}CDM}(k)$')
# ax1.set_xlabel(r'$k$[h/Mpc]')
# ax1.axhline(y=0, ls='dashed')

# # ax0.set_yscale('log')
# ax0.set_xscale('log')
# ax1.set_xscale('log')

# ax1.set_ylabel(r'emu/test - 1')

# ax0.plot(kvals, PmPl_all.T, alpha=0.15, color='k')


# ax0.set_xlim(kvals[0], kvals[-1])
# ax1.set_xlim(kvals[0], kvals[-1])
# # ax1.set_ylim(-9e-2, 9e-2)

# del_idx = [0, 1, 2, 5, 6, 7, 8]
# color_id = 0
# for x_id in del_idx:
#     print(x_id)
#     color_id = color_id + 1
#     time0 = time.time()
#     x_decodedGPy = Emu(parameter_array_all[x_id])  ## input parameters
#     time1 = time.time()
#     print('Time per emulation %0.4f' % (time1 - time0), ' s')
#     x_test = PmPl_all[x_id]

#     ax0.plot(kvals, x_decodedGPy, alpha=1.0, ls='--', lw = 1.9, dashes=(5, 5), label='emu', color=colorList[color_id])
#     ax0.plot(kvals, x_test, alpha=0.7, label='test', color=colorList[color_id])
#     ax1.plot( kvals, (x_decodedGPy[:]) / (x_test[:])  - 1, color=colorList[color_id])
    
# start, end = ax0.get_ylim()
# ax0.yaxis.set_ticks((np.arange(start, end, 0.1)))
# ax0.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))


# ax0.set_xticklabels([])
# plt.savefig(plotsDir + 'MGemu_rank' +str(nRankMax) + '.png', figsize=(28, 24), bbox_inches="tight")
# plt.show()


# In[17]:





# pca_recon1 = np.matmul(pca_weights, pca_bases)
# pca_recon = pca_model.inverse_transform(pca_weights)
# W_predArray, _ = GPy_predict(np.expand_dims(para_array, axis=0))

# test_idx = [0, 1, 2, 3]
# for x_id in test_idx:
# #     x_pca_recon = np.matmul(pca_weights, pca_bases)
#     para_array = parameter_array_all[x_id]
#     W_predArray, _ = GPy_predict(np.expand_dims(para_array, axis=0))
    
# #     x_emu_recon = pca_model.inverse_transform(W_predArray)
#     x_pca_recon = pca_model.inverse_transform(pca_weights)

#     x_test = PmPl_all[x_id]
    
# #     plt.plot(kvals, (x_emu_recon[0]) / (x_test) - 1, ls=':', color=plt.cm.Set1(color_id))
#     plt.plot(kvals, (x_pca_recon[x_id]) / (x_test) - 1, ls='--', dashes=(10, 2), color=plt.cm.Set1(color_id))

# #     plt.plot(kvals, (x_test) , ls='-', color=plt.cm.Set1(color_id))
# #     plt.plot(kvals, x_pca_recon[x_id], ls='--', dashes=(10, 2), color=plt.cm.Set1(color_id))
# #     plt.scatter((W_predArray[0]), pca_weights[x_id])

# plt.show()


# In[19]:


# pca_recon = np.matmul(pca_weights, pca_bases)
# x_pca_recon == pca_recon[x_id]
# x_pca_recon_ind = pca_model.inverse_transform(pca_weights[x_id])

# x_emu_ind = GPy_predict(np.expand_dims(para_array, axis=0)
# x_emu = GPy_predict(np.expand_dims(para_array, axis=0)

# x_emu_ind = Emu(np.expand_dims(parameter_array_all[x_id], axis=0) )
# x_emu = Emu(parameter_array_all)

# print('A-B', np.abs(x_pca_recon[x_id] - x_pca_recon_ind).max())
# print('A-true', np.abs(x_pca_recon[x_id] - PmPl_all[x_id]).max())
# print('B-true', np.abs(PmPl_all[x_id] - x_pca_recon_ind).max())
# print('emuA-true', np.abs(PmPl_all[x_id]/x_emu_ind - 1).max())
# print('emuB-true', np.abs(PmPl_all[x_id]/x_emu[:,x_id]- 1).max())

# plt.plot(kvals, (x_emu[:,x_id]) / (PmPl_all[x_id] ) - 1, ls='--', dashes=(10, 2), color=plt.cm.Set1(color_id))
# plt.plot(kvals, (x_emu_ind[0] / PmPl_all[x_id] ) - 1, ls='-', color=plt.cm.Set1(color_id))


# In[20]:


# del_idx


# In[ ]:


# Emu(np.expand_dims(parameter_array_all[x_id], axis=0) )


# In[44]:

p1min, p2min, p3min, p4min, p5min = parameter_array_all.min(axis = 0)
p1max, p2max, p3max, p4max, p5max = parameter_array_all.max(axis = 0)
p1mean, p2mean, p3mean, p4mean, p5mean = parameter_array_all.mean(axis = 0)


# from ipywidgets import interact
# import numpy as np

# from bokeh.io import push_notebook, show#, output_notebook
# from bokeh.plotting import figure
# # output_notebook()

# x = kvals
# y = Emu(parameter_array_all.mean(axis = 0))#/(Pk_mean) - 1 


# p = figure(title="emulated P(k) ratio", plot_height=500, plot_width=600, y_range=(0,0.6),
#            background_fill_color='white')

# r = p.line(x, y, color='black', line_width=1.5, alpha=0.8)



# def update(p1=p1mean , p2= p2mean, p3=p3mean, p4=p4mean, p5=p5mean):
# #     if   f == "sin": func = np.sin
# #     elif f == "cos": func = np.cos
#     r.data_source.data['y'] = Emu(np.array([p1, p2, p3, p4, p5]))#/Pk_mean - 1
#     push_notebook()
    

# show(p)#, notebook_handle=True)


# interact(update, p1=(p1min,p1max), p2=(p2min,p2max), p3=(p3min, p3max), p4 = (p4min, p4max), p5 = (p5min, p5max))


''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.

conda install -y bokeh
'''
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure

# # ## for notebook
# import bokeh.io
# bokeh.io.output_notebook()
# ##
# Set up data


# N = 200
# x = np.linspace(0, 4*np.pi, N)
# y = np.sin(x)

x = kvals
y = Emu(parameter_array_all.mean(axis = 0))#/(Pk_mean) - 1 

source = ColumnDataSource(data=dict(x=x, y=y))


plot = figure(title="emulated P(k) ratio", plot_height=600, plot_width=900, y_range=(0,0.6),
           background_fill_color='white')

# plot.line(x, y, color='black', line_width=1.5, alpha=0.8)




# Set up plot
# plot = figure(plot_height=400, plot_width=400, title="my sine wave",
#               tools="crosshair,pan,reset,save,wheel_zoom",
#               x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)




# def update(p1=p1mean , p2= p2mean, p3=p3mean, p4=p4mean, p5=p5mean):
# #     if   f == "sin": func = np.sin
# #     elif f == "cos": func = np.cos
#     r.data_source.data['y'] = Emu(np.array([p1, p2, p3, p4, p5]))#/Pk_mean - 1
#     push_notebook()
    

# show(p)#, notebook_handle=True)


# interact(update, p1=(p1min,p1max), p2=(p2min,p2max), p3=(p3min, p3max), p4 = (p4min, p4max), p5 = (p5min, p5max))




# Set up widgets
text = TextInput(title="Emu", value='P(k) ratio')
p1 = Slider(title=allLabels[0], value=p1mean, start=p1min, end=p1max, step=0.1)
p2 = Slider(title=allLabels[1], value=p2mean, start=p2min, end=p2max, step=0.1)
p3 = Slider(title=allLabels[2], value=p3mean, start=p3min, end=p3max, step=0.1)
p4 = Slider(title=allLabels[3], value=p4mean, start=p4min, end=p4max, step=0.1)
p5 = Slider(title=allLabels[4], value=p5mean, start=p5min, end=p5max, step=0.1)



# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    p1v = p1.value
    p2v = p2.value
    p3v = p3.value
    p4v = p4.value
    p5v = p5.value
    
    x = kvals
    y = Emu(np.array([p1v, p2v, p3v, p4v, p5v]))

    source.data = dict(x=x, y=y)
#     source.data = Emu(np.array([p1, p2, p3, p4, p5]))
    
   
    
# def update(p1=p1mean , p2= p2mean, p3=p3mean, p4=p4mean, p5=p5mean):
# #     if   f == "sin": func = np.sin
# #     elif f == "cos": func = np.cos
#     r.data_source.data['y'] = Emu(np.array([p1, p2, p3, p4, p5]))#/Pk_mean - 1
#     push_notebook()
    
    

for allP in [p1, p2, p3, p4, p5]:
    allP.on_change('value', update_data)


# Set up layouts and add to document
# inputs = column(text, offset, amplitude, phase, freq)

inputs = column(text, p1, p2, p3, p4, p5)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Emu"




