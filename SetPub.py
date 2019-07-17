import numpy as np
import matplotlib.pylab as plt
from matplotlib import ticker
np.set_printoptions(precision=3)


def set_pub():
    """ Pretty plotting changes in rc for publications
    Might be slower due to usetex=True
    
    
    plt.minorticks_on()  - activate  for minor ticks in each plot

    """
    plt.rc('font', weight='bold')    # bold fonts are easier to see
    #plt.rc('font',family='serif')
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)   # Slower
    plt.rc('font',size=12)    # 18 usually
    # plt.rcParams['image.cmap'] = 'nipy_spectral_r'
    plt.rcParams['image.cmap'] = 'plasma'


    plt.rc('lines', lw=1, color='k', markeredgewidth=1.5) # thicker black lines
    #plt.rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    plt.rc('savefig', dpi=300)       # higher res outputs
    
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    plt.rc('axes',labelsize= 10)
    
    plt.rcParams['xtick.major.size'] = 12
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['xtick.minor.size'] = 8
    plt.rcParams['xtick.minor.width'] = 1
    
    plt.rcParams['ytick.major.size'] = 12
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['ytick.minor.size'] = 8
    plt.rcParams['ytick.minor.width'] = 1
    
    #plt.rcParams.update({'font.size': 15})
    #plt.rcParams['axes.color_cycle'] = [ 'navy', 'forestgreen', 'darkred']


#import SetPub
#SetPub.set_pub()
