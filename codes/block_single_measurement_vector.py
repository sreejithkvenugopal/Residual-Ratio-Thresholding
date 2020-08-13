import numpy as np
import numpy.linalg as lin
import numpy.random as random
import scipy as sci
import matplotlib.pyplot as plt
from scipy import special
from scipy.linalg import hadamard
import os,sys
os.getcwd()
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.realpath('.'))

from codes.residual_ratio_thresholding import residual_ratio_thresholding

class block_single_measurement_vector():
    def __init__(self):
        self.norm_order=1
        pass
        
    def compute_signal_and_support(self,X,Y,block_size=4,algorithm='BOMP',alpha_list=[0.1]):
        self.Y=Y;self.X=X; nsamples,nfeatures=X.shape; self.block_size=block_size;
        self.kmax=np.min([nfeatures,np.int(np.floor(0.5*(nsamples+1)/block_size))]);
        self.nsamples=nsamples;self.nfeatures=nfeatures;self.alpha_list=alpha_list
        res_ratio,ordered_block_support_estimate_sequence=self.generate_residual_ratios(X,Y,algorithm)
        rrt=residual_ratio_thresholding(self.nsamples,self.nfeatures,alpha_list=self.alpha_list,
                                nchannels=1,group_size=self.block_size,scenario='compressed_sensing:BSMV')
        results=rrt.estimate_support(ordered_block_support_estimate_sequence,res_ratio)
        
        estimate_support_dict={}
        for alpha in alpha_list:
            est_alpha={}
            support_estimate=results[alpha]
            est_alpha['support_estimate']=support_estimate
            B_est= self.generate_estimate_from_support(X,Y,block_support=support_estimate)
            est_alpha['signal_estimate']=B_est
            estimate_support_dict[alpha]=est_alpha
        return estimate_support_dict
            
            
    def generate_residual_ratios(self,X,Y,algorithm):
         
        if algorithm=='BOMP':
            res_ratio,ordered_support_estimate_sequence=self.BOMP_run()
        else:
            # if you want to add a new function other than OMP and LASSO add that function here
            print('invalid algorithm')
        return res_ratio,ordered_support_estimate_sequence
    
    def generate_estimate_from_support(self,X,Y,block_support):
        support=self.generate_full_support_from_block_support(block_support)
        Xt=X[:,support]
        Beta_est=np.zeros((self.nfeatures,1))
        Beta_est[support]=np.matmul(lin.pinv(Xt),Y)
        return Beta_est


    def generate_full_support_from_block_support(self,block_support,block_size=None):
        if block_size is None:
            block_size=self.block_size
        ind = []
        for i in block_support:
            for j in range(i * block_size, (i + 1) * block_size):
                ind.append(j)
        return ind


    def correlated_block_selection(self,correlation,block_size=None):
        if block_size is None:
            block_size=self.block_size
        nfeatures=len(correlation)
        nblocks = np.int(nfeatures / block_size)
        block_norm = np.zeros(nblocks)
        for k in np.arange(nblocks):
            xt = []
            for j in range(k * block_size, (k + 1) * block_size):
                xt.append(correlation[j])
            block_norm[k] = lin.norm(np.array(xt), self.norm_order)
        sel_block = np.argmax(block_norm) #
        ind = [j for j in range(sel_block * block_size, (sel_block + 1) * block_size)] # indices corresponding the selected block
        ind_b = sel_block
        return ind_b, ind

    def BOMP_run(self):
        X=self.X;Y=self.Y;kmax=self.kmax; block_size=self.block_size;
        res_ratio=[]
        res_norm=[]
        res_norm.append(lin.norm(Y))
        res=Y # initialize the residual with observation
        ordered_support_estimate_sequence=[];ordered_block_support_estimate_sequence=[];
        flag=0;
        for k in range(kmax):
            correlation=np.matmul(X.T,res)
            selected_block,indices_in_block=self.correlated_block_selection(correlation)
            ordered_block_support_estimate_sequence.append(selected_block);
            ordered_support_estimate_sequence=ordered_support_estimate_sequence+indices_in_block
            Xk=X[:,ordered_support_estimate_sequence].reshape((self.nsamples,(k+1)*block_size))
            if lin.matrix_rank(Xk)<len(ordered_support_estimate_sequence) or lin.cond(Xk)>1e2 or np.max(correlation)<1e-8:
                print('Ill conditioned matrix. BOMP stop at iteration:'+str(k))
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
            while len(ordered_block_support_estimate_sequence)!=kmax:
                ordered_block_support_estimate_sequence.append(ordered_block_support_estimate_sequence[-1])
        return res_ratio,ordered_block_support_estimate_sequence
    
    def BOMP_prior_sparsity(self,X,Y,block_size,sparsity):
        res=Y # initialize the residual with observation
        nsamples=X.shape[0];nfeatures=X.shape[1];
        ordered_support_estimate_sequence=[];ordered_block_support_estimate_sequence=[];
        flag=0;
        for k in range(sparsity):
            correlation=np.matmul(X.T,res)
            selected_block, indices_in_block = self.correlated_block_selection(correlation)
            ordered_block_support_estimate_sequence.append(selected_block);
            ordered_support_estimate_sequence = ordered_support_estimate_sequence + indices_in_block
            Xk=X[:,ordered_support_estimate_sequence].reshape((self.nsamples,(k+1)*block_size))
            if lin.matrix_rank(Xk)<len(ordered_support_estimate_sequence) or lin.cond(Xk)>1e2 or np.max(np.abs(correlation))<1e-8:
                print('Ill conditioned matrix. BOMP stop at iteration:'+str(k))
                flag=1;break;
            else: 
                Xk_pinv=lin.pinv(Xk)
                Beta_est=np.zeros((nfeatures,1))
                Beta_est[ordered_support_estimate_sequence]=np.matmul(Xk_pinv,Y)
                res=Y-np.matmul(X,Beta_est)
        return ordered_block_support_estimate_sequence,Beta_est
    
    def generate_ordered_sequence(self,block_support_estimate_sequence,kmax=None):
        if kmax is None:
            kmax=self.kmax;
        nsupports=len(block_support_estimate_sequence)
        ordered_block_support_estimate_sequence=[]
        for k in np.arange(nsupports):
            if k==0:
                ordered_block_support_estimate_sequence=ordered_block_support_estimate_sequence+block_support_estimate_sequence[k]
            else:
                diff=[ele for ele in block_support_estimate_sequence[k]  if ele not in ordered_block_support_estimate_sequence]
                ordered_block_support_estimate_sequence=ordered_block_support_estimate_sequence+diff
        return ordered_block_support_estimate_sequence[:kmax]
    
    def res_ratios_from_ordered_sequence(self,ordered_block_support_estimate_sequence):
        X=self.X;
        Y=self.Y;
        kmax=len(ordered_block_support_estimate_sequence);
        res_ratio=[]
        res_norm=[]
        res_norm.append(lin.norm(Y))
        res=Y # initialize the residual with observation
        
        flag=0;
        for k in range(kmax):
            block_support=ordered_block_support_estimate_sequence[:(k+1)]
            indices_in_block=self.generate_full_support_from_block_support(block_support=block_support)
            Xk=X[:,indices_in_block].reshape((self.nsamples,len(indices_in_block)))
            if lin.matrix_rank(Xk)<len(indices_in_block) or lin.cond(Xk)>1e2 or res_norm[-1]<1e-10:
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
    
    

    
    def compute_error(self,block_support_true,block_support_estimate,Beta_true,Beta_estimate):
        Beta_true=np.squeeze(Beta_true); Beta_estimate=np.squeeze(Beta_estimate);
        block_support_true=set(block_support_true); block_support_estimate=set(block_support_estimate)
        l2_error=lin.norm(Beta_true-Beta_estimate)
        if block_support_true==block_support_estimate:
            support_error=0;
        else:
            support_error=1;
        return support_error,l2_error
    
    def generate_random_example(self,nsamples=50,nfeatures=100,block_size=4,sparsity=5,SNR_db=10):
        # sparsity is the number of blocks. actual sparsity=block_size*sparsity
        snr=10**(SNR_db/10)# SNR in real scale
        X=np.random.randn(nsamples,nfeatures)
        X=X/lin.norm(X,axis=0)
        Beta=np.zeros((nfeatures,1))

        if nfeatures % block_size != 0:
            raise Exception(' nfeatures should be a multiple of block_size')

        nblocks = np.int(nfeatures /block_size)
        block_support= np.random.choice(np.arange(nblocks), size=(sparsity), replace=False)
        full_support=self.generate_full_support_from_block_support(block_support,block_size=block_size)
        Beta[full_support] = np.sign(np.random.randn(len(full_support), 1))
        signal_power=len(full_support)
        noisevar = signal_power/ (nsamples * snr)
        noise = np.random.randn(nsamples, 1) * np.sqrt(noisevar)
        Y = np.matmul(X, Beta) + noise
        block_support=[np.int(x) for x in block_support]

        return X,Y,Beta,block_support,noisevar
