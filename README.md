# Residual-Ratio-Thresholding

Residual Ratio Thresholding  (RRT) is a technique to identify the the correct regression model from a sequence of  nested models.  

1). RRT can operate algorithms like LASSO, OMP and their derivatives without knowing signal sparsity and noise variance. 

2). RRT  provides easy to interpret final sample support recovery guarantees. 

3). RRT is closely related to various information theoretic criteria. However, unlike many of popular information theoretic criteria, RRT is based on finite sample distributional results. 

4). RRT is applicable to many different scenarios like group sparsity, row sparsity, robust regression with sparse outliers, linear model order selection etc. 

This codes are a result of joint work with Dr.Sheetal Kalyani, Dept. Of Electrical Engineering. (http://www.ee.iitm.ac.in/user/skalyani/)

The following papers are published based on the concept of Residual Ratio Thresholding. 


1. Signal and Noise Statistics Oblivious Orthogonal Matching Pursuit (ICML 2018, http://proceedings.mlr.press/v80/kallummil18a.html) (Operating OMP using RRT)

2. Noise Statistics Oblivious GARD For Robust Regression With Sparse Outliers (IEEE TSP 2018, https://ieeexplore.ieee.org/abstract/document/8543649) (Operating an algorithm form robust regression called GARD using RRT)

3). Residual Ratio Thresholding for Linear Model Order Selection (IEEE TSP  2019, https://ieeexplore.ieee.org/abstract/document/8573899) (Perform model order selection using RRT. Establish links between RRT and Information Theoretic criteria)

4). Generalized residual ratio thresholding (Under review in IEEE TSP 2020, https://arxiv.org/pdf/1912.08637.pdf) (Extended RRT to multiple measurement vectors, group sparsity, LASSO etc.)

5). High SNR consistent compressive sensing without noise statistics. (Elsevier Signal Processing 2020 https://www.sciencedirect.com/science/article/abs/pii/S0165168419303883). Developed a high SNR consistent version of RRT. 

Please cite the relevant papers while using this work. This set of Python RRT codes is a work under progress. Will be fully ready by November 2020. 

