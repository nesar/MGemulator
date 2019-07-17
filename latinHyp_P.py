''' Latin hypercube design
Installation: pip install --upgrade pyDOE
https://pythonhosted.org/pyDOE/randomized.html

'''

import numpy as np
from matplotlib import pyplot as plt
import pyDOE as pyDOE

def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


num_evals = 32  ## Total number of evaluations for training the emulator

np.random.seed(7)

#########################################################################
####### Parameters -- replace this and the limits ########
# para = np.linspace(lower_lim, upper_lim, total_eval)

para1 = np.linspace(0.10, 0.140, num_evals)  #OmegaM
para2 = np.linspace(0.0205, 0.0235, num_evals) #Omegab
para3 = np.linspace(0.7, 0.9, num_evals) # sigma8
para4 = np.linspace(0.55, 0.85, num_evals) # h
para5 = np.linspace(0.85, 1.05, num_evals) # n_s

para6 = np.linspace(0.5, 1.5, num_evals) # z_m
para7 = np.linspace(0.05, 0.5, num_evals) # FWHM


#########################################################################

AllPara = np.vstack([para1, para2, para3, para4, para5, para6, para7])

lhd = pyDOE.lhs(AllPara.shape[0], samples=num_evals, criterion=None) # c cm corr m
print(lhd)
print
# lhd = norm(loc=0, scale=1).ppf(lhd)  # this applies to both factors here

##
f, a = plt.subplots(AllPara.shape[0], AllPara.shape[0], sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.rcParams.update({'font.size': 8})

for i in range(AllPara.shape[0]):
    for j in range(i+1):
        print(i,j)
        if(i!=j):
            a[i, j].scatter(lhd[:, i], lhd[:, j], s=5)
            a[i, j].grid(True)
        else:
            # a[i,i].set_title(AllLabels[i])
            # a[i, i].text(0.4, 0.4, AllLabels[i], size = 'xx-large')
            hist, bin_edges = np.histogram(lhd[:,i], density=True, bins=64)
            # a[i,i].bar(hist)
            a[i,i].bar(bin_edges[:-1], hist/hist.max(), width=0.2)
            plt.xlim(0,1)
            plt.ylim(0,1)


#plt.savefig('../Cl_data/Plots/LatinSq.png', figsize=(10, 10))
plt.show()


idx = (lhd * num_evals).astype(int)

AllCombinations = np.zeros((num_evals, AllPara.shape[0]))
for i in range(AllPara.shape[0]):
    AllCombinations[:, i] = AllPara[i][idx[:, i]]

np.savetxt('lhc_'+str(num_evals)+'.txt', AllCombinations)   #### no

print(AllCombinations)