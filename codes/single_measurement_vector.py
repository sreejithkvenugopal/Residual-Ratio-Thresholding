import numpy as np
import numpy.linalg as lin
import numpy.random as random
import scipy as sci
import matplotlib.pyplot as plt
from scipy import special
from scipy.linalg import hadamard
import os
os.getcwd()
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')


class single_measurement_vector():
    def __init__(self):
        self.nothing=None
        
    def generate_residual_ratios(self,X,Y,algorithm):
        self.Y=Y;self.X=X; nsamples,nfeatures=X.shape; 
        self.kmax=np.min([nfeatures,np.int(np.floor(0.5*(nsamples+1)))]);
        self.nsamples=nsamples;self.nfeatures=nfeatures; 
        if algorithm=='OMP':
            res_ratio,ordered_support_estimate_sequence=self.OMP_run()
        elif algorithm=='LASSO':
            res_ratio,ordered_support_estimate_sequence=self.LASSO_run()
        else: 
            print('invalid algorithm')
        return res_ratio,ordered_support_estimate_sequence
    def generate_estimate_from_support(self,X,Y,support):
        Xt=X[:,support]
        Beta_est=np.zeros((self.nfeatures,1))
        Beta_est[support]=np.matmul(lin.pinv(Xt),Y)
        return Beta_est
    def OMP_run(self):
        X=self.X;Y=self.Y;kmax=self.kmax; 
        res_ratio=[]
        res_norm=[]
        res_norm.append(lin.norm(Y))
        res=Y # initialize the residual with observation
        ordered_support_estimate_sequence=[]
        flag=0;
        for k in range(kmax):
            correlation=np.abs(np.matmul(X.T,res))
            ind=np.argmax(correlation)
            ordered_support_estimate_sequence.append(ind)
            Xk=X[:,ordered_support_estimate_sequence].reshape((self.nsamples,k+1))
            if lin.matrix_rank(Xk)<len(ordered_support_estimate_sequence) or lin.cond(Xk)>1e2 or np.max(correlation)<1e-8:
                print('Ill conditioned matrix. OMP stop at iteration:'+str(k))
                flag=1;break;
            else: 
                Xk_pinv=lin.pinv(Xk)
                Beta_est=np.zeros((self.nfeatures,1))
                Beta_est[ordered_support_estimate_sequence]=np.matmul(Xk_pinv,Y)
                res=Y-np.matmul(X,Beta_est)
                res_normk=lin.norm(res)
                res_ratio.append(res_normk/res_norm[-1])
                res_norm.append(res_normk)
                
        if flag==1:
            while len(res_ratio)!=kmax:
                res_ratio.append(1)
            while len(ordered_support_estimate_sequence)!=kmax:
                ordered_support_estimate_sequence.append(ordered_support_estimate_sequence[-1])
        return res_ratio,ordered_support_estimate_sequence
    
    def OMP_prior_sparsity(self,X,Y,sparsity):
        res=Y # initialize the residual with observation
        support_estimate=[]
        flag=0;
        for k in range(sparsity):
            correlation=np.abs(np.matmul(X.T,res))
            ind=np.argmax(correlation)
            support_estimate.append(ind)
            Xk=X[:,support_estimate].reshape((self.nsamples,k+1))
            if lin.matrix_rank(Xk)<len(support_estimate) or lin.cond(Xk)>1e2 or np.max(correlation)<1e-8:
                print('Ill conditioned matrix. OMP stop at iteration:'+str(k))
                flag=1;break;
            else: 
                Xk_pinv=lin.pinv(Xk)
                Beta_est=np.zeros((self.nfeatures,1))
                Beta_est[support_estimate]=np.matmul(Xk_pinv,Y)
                res=Y-np.matmul(X,Beta_est)
        return support_estimate,Beta_est
    
    def generate_ordered_sequence(self,support_estimate_sequence,kmax=None):
        if kmax is None:
            kmax=self.kmax;
        nsupports=len(support_estimate_sequence)
        ordered_support_estimate_sequence=[]
        for k in np.arange(nsupports):
            if k==0:
                ordered_support_estimate_sequence=ordered_support_estimate_sequence+support_estimate_sequence[k]
            else:
                diff=[ele for ele in support_estimate_sequence[k]  if ele not in ordered_support_estimate_sequence]
                ordered_support_estimate_sequence=ordered_support_estimate_sequence+diff
        return ordered_support_estimate_sequence[:kmax]
    
    def res_ratios_from_ordered_sequence(self,ordered_support_estimate_sequence,X=None,Y=None):
        if X is None:
            X=self.X;
        if Y is None:
            Y=self.Y;
        kmax=len(ordered_support_estimate_sequence);
        res_ratio=[]
        res_norm=[]
        res_norm.append(lin.norm(Y))
        res=Y # initialize the residual with observation
        
        flag=0;
        for k in range(kmax):
            index=ordered_support_estimate_sequence[:(k+1)]
            Xk=X[:,index].reshape((self.nsamples,len(index)))
            if lin.matrix_rank(Xk)<len(index) or lin.cond(Xk)>1e2 or res_norm[-1]<1e-10:
                print('Ill conditioned matrix. stop at iteration:'+str(k))
                flag=1;break;
            else: 
                Xk_pinv=lin.pinv(Xk)
                Beta_est=np.zeros((self.nfeatures,1))
                Beta_est[index]=np.matmul(Xk_pinv,Y)
                res=Y-np.matmul(X,Beta_est)
                res_normk=lin.norm(res)
                res_ratio.append(res_normk/res_norm[-1])
                res_norm.append(res_normk)
                
        if flag==1:
            while len(res_ratio)!=kmax:
                res_ratio.append(1)
            
        return res_ratio
    
    
    def LASSO_run(self):
        X=self.X;Y=self.Y
        _, _, coefs = linear_model.lars_path(X, Y.reshape((self.nsamples,)), method='lasso', verbose=False)
        nfeatures,nodes=coefs.shape
        support_estimate_sequence=[]
        for k in np.arange(nodes):
            support_k=np.where(np.abs(coefs[:,k])>1e-8)[0]
            support_estimate_sequence.append(support_k.tolist())
        ordered_support_estimate_sequence=self.generate_ordered_sequence(support_estimate_sequence)
        res_ratio=self.res_ratios_from_ordered_sequence(ordered_support_estimate_sequence)
        return res_ratio,ordered_support_estimate_sequence
    
    def compute_error(self,support_true,support_estimate,Beta_true,Beta_estimate):
        Beta_true=np.squeeze(Beta_true); Beta_estimate=np.squeeze(Beta_estimate);
        support_true=set(support_true); support_estimate=set(support_estimate)
        l2_error=lin.norm(Beta_true-Beta_estimate)
        if support_true==support_estimate:
            support_error=0;
        else:
            support_error=1;
        return support_error,l2_error
    
    def generate_random_example(self,nsamples=50,nfeatures=100,sparsity=5,SNR_db=10):
        X=np.random.randn(nsamples,nfeatures)
        X=X/lin.norm(X,axis=0)
        Beta=np.zeros((nfeatures,1))
        Beta[:sparsity]=np.sign(np.random.randn(sparsity,1))
        signal=np.matmul(X,Beta)
        signal_power=sparsity;
        snr=10**(SNR_db/10)
        noisevar=signal_power/(nsamples*snr)
        
        Y=signal+np.random.randn(nsamples,1)*np.sqrt(noisevar)
        
        support=[np.int(k) for k in np.arange(sparsity)]
        
        return X,Y,Beta,support,noisevar
