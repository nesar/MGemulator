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
    # conda install -c r rpy2
    
    Built by N. Ramachandra and M. Binois, HEP and MCS divisions, Argonne National Laboratories.
    """

import numpy as np

###### R kernel imports from rpy2 #####
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r
from rpy2.robjects.packages import importr



########## R imports ############
RcppCNPy = importr('RcppCNPy')
# RcppCNPy.chooseCRANmirror(ind=1) # select the first mirror in the list
## There are other importr calls in PCA and GP functions


########################### PCA ###################################
def PCA_decomp(nRank = 8):
    Dicekriging = importr('DiceKriging')
    r('require(foreach)')
    ro.r.assign("nrankmax", nRank)
    r('svd(y_train2)')
    r('svd_decomp2 <- svd(y_train2)')
    r('svd_weights2 <- svd_decomp2$u[, 1:nrankmax] %*% diag(svd_decomp2$d[1:nrankmax])')


######################## GP PREDICTION ###############################

def GP_model_load(model):
    GPareto = importr('GPareto')
    
    ro.r('''
        
        GPmodel <- gsub("to", "",''' + model + ''')
            
            ''')
    
    r('''if(file.exists(GPmodel)){
        load(GPmodel)
        }else{
        print("ERROR: No trained GP file")
        }''')

def GP_predict(para_array):
    GPareto = importr('GPareto')
    
    
    para_array = np.expand_dims(para_array, axis=0)
    nr, nc = para_array.shape
    Br = ro.r.matrix(para_array, nrow=nr, ncol=nc)
    
    ro.r.assign("Br", Br)
    
    r('wtestsvd2 <- predict_kms(models_svd2, newdata = Br , type = "UK")')
    r('reconst_s2 <- t(wtestsvd2$mean) %*% t(svd_decomp2$v[,1:nrankmax])')
    
    y_recon = np.array(r('reconst_s2'))
    
    return y_recon[0]

############################################################
