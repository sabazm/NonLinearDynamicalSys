#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:35:36 2022

@author: sabazamankhani
"""



import scipy.stats as spp
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky
import scipy as scp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
import scipy.special as sp
import scipy.stats as stats
import scipy.linalg as linalg
from pandas.plotting import register_matplotlib_converters
from scipy.stats import norm
import scipy.stats as ss
register_matplotlib_converters()
np.seterr(divide='ignore', invalid='ignore')
sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams['figure.dpi'] = 100
np.random.seed(1)

#float_formatter = "{:.5f}".format
#np.set_printoptions(suppress=True, precision=5)
plt.style.use('bmh')

def gibbs_param( Phi, Psi, Sigma, V, Lambda, l, T ):

    '''Generate parameters from the full conditional posterior distribution

    Args:\n
    Phi (float): nx x nx symmetric positive definite matrix
    Psi (float): nx x 1
    Sigma (float): nb x nb symmetric positive definite matrix
    V (float): nb x nb symmetric positive definite matrix
    Lambda (float): nx x nx symmetric positive definite matrix
    l (int): hyperparameter
    T (int): number of observations

    Retuns:
    A (float): nx x nb matrix
    Q (float): nx x nx symmetric positive definite matrix\n
    '''

    nb = V.shape[0]
    nx = np.size(Phi)
    
    M = np.zeros((nx,nb))
    Phibar = Phi + ((M @ np.linalg.inv(V))@M.T)
    Psibar = Psi + (M @ np.linalg.inv(V))
    Sigbar = Sigma + np.linalg.inv(V)
    cov_M = Lambda+Phibar-((Psibar @ np.linalg.inv(Sigbar)) @ Psibar.T)
    cov_M_sym = 0.5*(cov_M + cov_M.T) 
    Q = ss.invwishart.rvs(scale=cov_M_sym,df=T+l)
    #Q = np.array([[0.0009]])
    #X = ss.norm.ppf(np.random.rand(nx,nb))
    #X = np.ones((nx,nb))*0.01
    X = (np.random.normal(loc=0,scale=1,size=(nx,nb)))
    post_mean = Psibar @ np.linalg.inv(Sigbar)
    A = post_mean + scp.linalg.cholesky(Q)@ X @ scp.linalg.cholesky(np.linalg.inv(Sigbar))
    return A, Q


#Phi: cross correlation or covariance of the residual with itself over the whole time dimension for a given dimension, in this case either x_1 or x_2.
#Psi: cross correlation or covariance of the residual with functions of the unknown states in a given dimension. In the example these are coupled like [x_1_t' u_t'] (term in f_1) and [x_2_t' x_1_t' u_t'] (term in f_2).
#Sig: cross correlation or covariance of the unknown functions of the states over the whole time
#V: covariance of the state noise, Block diagonal with unknown states in each dimension. In the example we only have [x_1_t] and [x_1_t x_2_t]
#LambdaQ: needed to compute posterior of Q, in this setup we always use LambdaQ = lQ*eye(block size). This is because we are interested in inference on Q and want the posterior to be able to recover the true state-noise when specified instead of priors only for the off-diagonal elements.
#lQ: prior state noise that is used to compute LambdaQ
#time: length of the time signals. This is necessary because we compute some statistics only over a finite time. 
#and the outputs are:
##A: the unknown parameters. For example, if [f_1 f_2]' = [x_1_t' u_t']*A_1 + [x_2_t' x_1_t' u_t']*A_2, then A is [A_1 A_2] and it can be sampled with Gibbs sampling. The A prior comes from states and functions that have been previously sampled, scaled with lambdas, and turned into pseudo-observations (called Psi in this application).
##Q: the state noise. The prior for Q comes from the model's own previous state noise. This is called Sig in this application.
##n: number of discontinuity points that are used in one dimension. Note that in this script only f_1 has discontinuity points for 1D functions. f_2 has an a priori included discontinuity because
##that state is decoupled most of the time. This can of course be changed in the script. 
##pts: the actual discontinuity points, they are sampled in the script, and then re-used here. 
##LambdaQ: another copy of the parameter LambdaQ. It is computed in the gibbs_meas function and not passed as input, but it is given as output so that LambdaQ it can be displayed and tracked. The reason why it is re-computed in the script instead of using the input LambdaQ is that here we have a small time difference between the old and the new time point.


# def systematic_resampling(W,N):

#     #idx=np.array([])
#     #N=len(W)
#     if N==1:
#         indexes= np.array([np.random.randint(0,30)])
#     else:
#         positions = (np.random.random()+np.arange(N))/N
    
#         indexes = np.zeros(N,'i')
#         cumulative_sum = np.cumsum(W)
#         i, j = 0,0
#         while i < N:
#             if positions[i] < cumulative_sum[j]:
#                 indexes[i] = j
#                 i +=1
#             else:
#                 j+=1
#     return indexes
 
def systematic_resampling(weights, N):
    
    W = weights/np.sum(weights)
    u = 1/N*np.random.rand()

    idx = np.zeros(N)
    q = 0
    n = 0
    for i in range(N):
        while q < u:
            n += 1
            q = q + W[n-1]
        idx[i] = n-1
        u = u + 1/N
    return idx

# def systematic_resampling(weights,N):
#     pos = (np.arange(N) + np.random.uniform()) / N
#     indexes = np.zeros(N, 'i')
#     cumulative_sum = np.cumsum(weights)
#     i, j = 0, 0
#     while i < N:
#         if pos[i] < cumulative_sum[j]:
#             indexes[i] = j
#             i += 1
#         else:
#             j += 1
#     return indexes
# def systematic_resampling(weights, N):
# #''' u is sampled from a uniform distribution, then W is normalized and sorted, then the code goes through each element in W and
# #       adds it until the sum is greater than u.
# #       When a match is found the index of that match is added to idx.
# #       Then u is updated and the process repeats.'''
#     indexes = []
#     u = np.random.rand(N)
#     w = weights / sum(weights)
#     w_cum = np.cumsum(w)
#     idx = 0
#     for n in range(len(u)):
#         while w_cum[idx] < u[n]:
#             idx += 1
#         indexes.append(idx)
#     return np.array(indexes)



# def compute_marginal_likelihood(Phi, Psi, Sigma, V, Lambda, l, N):
#     nb = V.shape[0]
#     nx = Phi.shape[0]
    
#     M = np.zeros((nx,nb))
    
#     Phibar = Phi + np.dot(np.dot(M,np.linalg.inv(V)),M.T)
#     Psibar = Psi + np.dot(M,np.linalg.inv(V))
#     Sigbar = Sigma + np.linalg.inv(V)
    
#     Lambda_post = Lambda+Phibar-np.dot(np.dot(Psibar,np.linalg.inv(Sigbar)),Psibar.T)
#     l_post = l+N
    
#     gamma_lnx_prior     = np.log(np.pi)*(((nx)-1)*(nx)/4) + np.sum(sp.loggamma(l/2 + (1 -np.arange(1,nx+1))/2))
#     gamma_lnx_posterior = np.log(np.pi)*(((nx)-1)*(nx)/4) + np.sum(sp.loggamma(l_post/2 + (1 -np.arange(1,nx+1))/2))
    
#     marg_log_lik_fr_lik   = np.log(2*np.pi)*(-N/2)
#     marg_log_lik_fr_post  = -np.log(2)*nx*l_post/2 - gamma_lnx_posterior + np.log(np.linalg.det(Lambda_post))*l_post/2 - np.log(2*np.pi)*nx*nb/2 + np.log(np.linalg.det(Sigbar))*nx/2
#     marg_log_lik_fr_prior = -np.log(2)*nx*l/2      - gamma_lnx_prior     + np.log(np.linalg.det(Lambda))**l/2           - np.log(2*np.pi)*nx*nb/2 - np.log(np.linalg.det(V))*nx/2
   
#     marg_lok_lik = marg_log_lik_fr_lik + marg_log_lik_fr_prior - marg_log_lik_fr_post
  
#     return marg_lok_lik

####################################################
#This funtion calculates the computed marginal likelihood from the 
        #  VAR model model, i.e.

        # p(y|x_0) = \int p(y|x,x_0) p(x) \d x

        # where
        #  y = Prediction horizon T observations, 
        #  X_t = n_x number of state variables T observations, 
        #  X_0 = Prior mean and variance

        # #HOW TO USE: 
        # compute_marginal_likelihood(Phi, Psi, Sigma, V, Lambda, l, N)

        # Inputs:
        # Phi      = n_x x n_x, VAR matrix: phi(1)*x_t +...+phi(p)*x(t-p) = y_t
        # Psi      = n_x, x n_b, VAR matrix: psi(1)* beta_1 + ... + psi(p)* beta_p = y_t
        # Sigma    = n_x x n_x, innovation variance
        # V        = n_b x n_b, inverse Wishart prior variance
        # l        = Scalar, Prior degree of freedom
        # N        = Scalar, Number of observations in the sample

        # Outputs:
        # marg_lok_lik = Scalar, The log of the marginal likelihood


def compute_marginal_likelihood(Phi, Psi, Sigma, V, Lambda, l, N):
    
    nb = V.shape[0]
    nx = Phi.shape[0]
    
    M = np.zeros((nx,nb))
    
    Phibar = Phi + (M@np.linalg.inv(V))@M.T
    Psibar = Psi + M@np.linalg.inv(V)
    Sigbar = Sigma + np.linalg.inv(V)
    
    Lambda_post = Lambda + Phibar - (Psibar@np.linalg.inv(Sigbar))@Psibar.T
    l_post = l + N
    
    gamma_lnx_prior     = np.log(np.pi)*(((nx)-1)*(nx)/4) + np.sum(sp.loggamma(l/2 + (1 - np.arange(1,nx+1))/2))
    gamma_lnx_posterior = np.log(np.pi)*(((nx)-1)*(nx)/4) + np.sum(sp.loggamma(l_post/2 + (1 - np.arange(1,nx+1))/2))
    marg_log_lik_fr_lik   = np.log(2*np.pi)*(-N/2)
    marg_log_lik_fr_post  = -np.log(2)*nx*l_post/2 - gamma_lnx_posterior + np.log(np.linalg.det(Lambda_post))*l_post/2 - np.log(2*np.pi)*nx*nb/2 + np.log(np.linalg.det(Sigbar))*nx/2
    marg_log_lik_fr_prior = -np.log(2)*nx*l/2      - gamma_lnx_prior     + np.log(np.linalg.det(Lambda))**l/2           - np.log(2*np.pi)*nx*nb/2 - np.log(np.linalg.det(V))*nx/2
   
    marg_lok_lik = marg_log_lik_fr_lik + marg_log_lik_fr_prior - marg_log_lik_fr_post
    
    return marg_lok_lik
 

np.random.seed(1)

dta = pd.read_csv('dataBenchmark.csv')
u= np.expand_dims(dta['uEst'].values,axis=0)
y= np.expand_dims(dta['yEst'].values,axis=0)
plt.plot(u[0])
plt.plot(y[0])
plt.show()

u_mean = u.mean()
y_mean = y.mean()

def normalize(data):
    mean =np.round(data.mean(),5)
    return data-mean

u = normalize(u[:])

#u_mean = u.mean()
y = normalize(y[:])
#y_mean = y.mean()
T = u.shape[1]

iA =np.array([[0.8041,0.0422],[-0.0327,0.9890]])
iB = np.array([[-0.7000],[-0.0382]])
iC = np.array([[-1.11022302462516e-16, 1.]])


def g_i(x,u=None):
    return np.array([0,1])@(x)

R = 0.01
nx = 2 
nu = 1 
ny = 1


# Parameters for the algorithm, priors, and basis functions
K = 10000
N = 30

# Basis functions for f:
n_basis_u = 5
n_basis_x1 = 5
n_basis_x2 = 5
#L = np.zeros((1,1,3))

L = np.expand_dims([5.,15.,6.],axis=0)


n_basis_1 = n_basis_u * n_basis_x1
jv_1 = np.zeros((n_basis_1,nx-1+nu))
lambda_1 = np.zeros((n_basis_1,nx-1+nu))


n_basis_2 = n_basis_u * n_basis_x1 * n_basis_x2
jv_2 = np.zeros((n_basis_2,nx+nu))
lambda_2 = np.zeros((n_basis_2,nx+nu))


# 2D (f_1)
for i in range(1,n_basis_u+1):
    for j in range(1, n_basis_x1+1):
        ind = n_basis_x1*(i-1) + j
        jv_1[ind-1,:] = [i,j]
        lambda_1[ind-1,:] = (np.pi *np.array([ i,j]).T/(2*L[:,0:2]))**2


# 3D (f_2)
for i in range(1,n_basis_u+1):
    for j in range(1,n_basis_x1+1):
        for k in range(1,n_basis_x2+1):
            ind = n_basis_x1*n_basis_x2*(i-1) + n_basis_x2*(j-1) + k
            jv_2[ind-1,:] = [i,j,k]
            lambda_2[ind-1,:] = (np.pi*np.array([i,j,k]).T/(2*L[:,:]))**2

jv_1 =np.expand_dims(jv_1,axis=1)
jv_2 = np.expand_dims(jv_2,axis=1)


#phi_1 = lambda x1,u: np.prod(np.multiply(L[:,:2]**(-1/2),np.sin(np.multiply(np.multiply((np.pi*jv_1),(np.expand_dims(np.vstack((u,x1)).T,axis=0)+L[np.zeros((x1.shape[0])).astype(int),:2])),1.0/2*L[:,:2]))),2)
#phi_2 = lambda x,u:np.prod(np.multiply(L**(-1/2),np.sin(np.multiply(np.multiply((np.pi*jv_2),((np.expand_dims(np.vstack((u,x)).T,axis=0))+L[np.zeros((x.shape[1])).astype(int),:])),1.0/2*L))),2)
def phi_1(x1,u):
    sumux1l = np.expand_dims(np.vstack((u,x1)).T,axis=0)+L[np.zeros((x1.shape[0])).astype(int),:2]
    mul1= np.multiply((np.pi*jv_1),sumux1l)
    mul2=np.multiply(mul1,1.0/2*L[:,:2])
    sinmul= np.sin(mul2)
    mul3 = np.multiply(L[:,:2]**(-1/2),sinmul)
    return np.prod(mul3,2)
def phi_2(x,u):
    sumuxl = np.expand_dims(np.vstack((u,x)).T,axis=0)+L[np.zeros((x.shape[1])).astype(int),:]
    mul1 = np.multiply((np.pi*jv_2),sumuxl)
    mul2 = np.multiply(mul1,1.0/2*L)
    sinmul= np.sin(mul2)
    mul3 = np.multiply(L**(-1/2),sinmul)
    return np.prod(mul3,2)
    

#phi_1 = lambda x1,u: np.prod(L[:,0:2]**(-1/2)*np.sin(np.pi*jv_1*(np.array([u,x1])+L[:,0:2])/(2*L[:,0:2])),axis=1)
#phi_2 = lambda x,u: np.prod(L**(-1/2)*np.sin(np.pi*jv_2*(np.array([u,x])+L)/(2*L)),axis=1)

# GP prior
S_SE = lambda w, ell : np.sqrt(2*np.pi*ell**2)*np.exp(-(np.square(w)/2)*ell**2)
#V1 = lambda n1 : 100*np.diag(np.tile(np.prod(S_SE(np.sqrt(lambda_1),np.tile(np.array([3,3]),(n_basis_1,1))),axis=1),1))
V1 = lambda n1: 100*np.diag(np.squeeze(np.tile(np.prod(S_SE(np.sqrt(lambda_1),np.tile(np.array([3,3]),(n_basis_1,1))),keepdims=True,axis=1),(n1+1,1))))
V2 = lambda n2: 100*np.diag(np.squeeze(np.tile(np.prod(S_SE(np.sqrt(lambda_2),np.tile(np.array([3,3,3]),(n_basis_2,1))),keepdims=True,axis=1),(n2+1,1))))

# Priors for Q
lQ1 = 1000
lQ2 = 1000
LambdaQ1 = 1*np.eye(1)
LambdaQ2 = 1*np.eye(1)

model_state_1 = [{} for i in range(K+2)]
model_state_2 = [{} for i in range(K+2)]

model_state_1[0]['A'] = np.zeros((1,n_basis_1))
model_state_1[0]['Q'] = 1
model_state_1[0]['n'] = 0
model_state_1[0]['pts'] = np.array([-L[0][1],L[0][1]])

model_state_2[0]['A'] = np.zeros((1,2*n_basis_2))
model_state_2[0]['Q'] = 1
model_state_2[0]['n'] = 1
model_state_2[0]['pts'] = np.array([-L[0][2],4.4,L[0][2]])
####################
def f_i(x,u):
    f_1 = (iA[0:1,:]@x)+ (iB[0:1]@u) + (model_state_1p[0]@phi_1(x[0:1,:],u))
    f_2 = (iA[1:2,:]@x) + (iB[1:2]@u) + (model_state_2p[0]@phi_2(x,u))
    return np.vstack((f_1,f_2))


dot = lambda x,y: sum([a*b for a,b in zip(x,y)])

#f_i = lambda x,u: np.array([np.dot(iA[0,:],x)+ np.dot(iB[0,:],u) + np.squeeze(np.dot(model_state_1p[0],phi_1(x[0,:],u))), np.dot(iA[1,:],x) + np.dot(iB[1,:],u) + np.squeeze(np.dot(model_state_2p[0],phi_2(x,u)))])

ys = np.zeros((3,T))
for i in range(3):

    model_state_1p = gibbs_param(0, 0, 0, V1(0), LambdaQ1,lQ1,0)
    model_state_2p = gibbs_param(0, 0, 0, V2(0), LambdaQ2,lQ2,0)
    
        

    #f_i = lambda x,u: np.array([np.dot(iA[0,:],x)+ np.dot(iB[0,:],u) + np.dot(model_state_1p[0],phi_1(x[0,:],u)), np.dot(iA[1,:],x) + np.dot(iB[1,:],u) + np.dot(model_state_2p[0],phi_2(x,u))])
    #f_i = lambda x,u: np.vstack([np.dot(iA[0,:],x)+ np.dot(iB[0],u) + np.dot(model_state_1p[0],phi_1(x[0,:],u)) , np.dot(iA[1,:],x) + np.dot(iB[1],u) + np.dot(model_state_2p[0],phi_2(x,u))])

    xs = np.zeros((2,1))
    

    for t in range(T):
        ys[i,t]=(g_i(xs,0))
        xs = f_i(xs,u[:,t])

    plt.plot(ys[i,:])

    plt.draw()
    plt.pause(0.01)

plt.show()





##############################
phi_1_ = lambda x1,u: np.prod(np.multiply(L[:,:2]**(-1/2),np.sin(np.multiply(np.multiply((np.pi*jv_1),((np.transpose(np.vstack((u,x1))[:,:,None], (2,1,0)))+L[np.zeros((x1.shape[0])).astype(int),:2])),1.0/2*L[:,:2]))),2)
phi_2_ = lambda x,u:np.prod(np.multiply(L**(-1/2),np.sin(np.multiply(np.multiply((np.pi*jv_2),((np.transpose(np.vstack((u,x))[:,:,None], (2,1,0)))+L[np.zeros((x.shape[1])).astype(int),:])),1.0/2*L))),2)


p1 = 0.9


# Pre-allocate
x_prim = np.zeros((nx,1,T))

############################
#MCMC Algorithm
############################
## Run learning algorithm


#The code is running a particle filter with ancestor sampling (CPF-AS) to sample a trajectory from the posterior distribution of the state given the measurements. It then uses this sample to update the parameters of the model. 
#The model is a simple linear system with a discontinuity in the state transition function. The discontinuity is modelled using a mixture of Gaussians. The mixture is parametrized by the number of components, the means and the variances of each component. The number of components is updated using a Gibbs sampler. The means and variances are updated using a Gibbs sampler. 

Mar_lik=[]

for k in range(K+1):
    
    Qi = np.zeros((nx,nx))
    
    pts1 = model_state_1[k]['pts']
    n1 = model_state_1[k]['n']
    Ai1 = model_state_1[k]['A']
    Qi[0,0] = model_state_1[k]['Q']
    pts2 = model_state_2[k]['pts']
    n2 = model_state_2[k]['n']
    Ai2 = model_state_2[k]['A']
    Qi[1,1] = model_state_2[k]['Q']
    
    
    def f_i(x,u):
        
        phi_1_tile = np.tile(phi_1_(x[0,:],u[np.zeros((1,x.shape[1])).astype(int)]),(n1+1,1))
        less_1 = np.tile(x[0,:],(n_basis_1*(n1+1),1))<np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))
        greater_1 = np.tile(x[0,:],(n_basis_1*(n1+1),1))>=np.kron(np.expand_dims(pts1[:-1],axis=0).T,np.ones((n_basis_1,1)))
        f_1 = Ai1 @ np.multiply(greater_1,np.multiply(less_1,phi_1_tile))
        
        phi_2_tile = np.tile(phi_2_(x,u[np.zeros((1,x.shape[1])).astype(int)]),(n2+1,1))
        less_2 = np.tile(x[1,:],(n_basis_2*(n2+1),1))<np.kron(np.expand_dims(pts2[1:],axis=0).T,np.ones((n_basis_2,1)))
        greater_2 = np.tile(x[1,:],(n_basis_2*(n2+1),1))>=np.kron(np.expand_dims(pts2[:-1],axis=0).T,np.ones((n_basis_2,1)))
        f_2 = Ai2 @ np.multiply(greater_2,np.multiply(less_2,phi_2_tile))
        return iA@x+iB@u[np.zeros((1,x.shape[1])).astype(int)] + np.vstack((f_1,f_2))
    
 
    
    Q_chol = np.linalg.cholesky(Qi)
    
    # Pre-allocate
    w = np.zeros((T,N))
    x_pf = np.zeros((nx,N,T))
    a = np.zeros((T,N))

    # Initialize
    if k > 0: 
        x_pf[:,-1:,:] = x_prim
    
    w[0,:] = 1
    w[0,:] = w[0,:]/np.sum(w[0,:])
    
    # CPF with ancestor sampling
    
    x_pf[:,:-1,0] = 0
    for t in range(T):
        # PF time propagation, resampling and ancestor sampling
        if t >= 1:
            if k > 0:
                a[t,:N-1] = systematic_resampling(w[t-1,:],N-1)
                x_pf[:,:N-1,t] = f_i(x_pf[:,a[t,0:N-1].astype(int),t-1],u[:,t-1]) + Q_chol @ np.random.randn(nx,N-1)

                waN = np.multiply(w[t-1,:],spp.multivariate_normal.pdf(f_i(x_pf[:,:,t-1],u[:,t-1]).T,x_pf[:,N-1,t].T,Qi))
                waN = waN/np.sum(waN)
                a[t,N-1] = systematic_resampling(waN,1)
            else: # Run a standard PF on first iteration
                a[t,:] = systematic_resampling(w[t-1,:],N)
                x_pf[:,:,t] = f_i(x_pf[:,a[t,:].astype(int),t-1],u[:,t-1]) + Q_chol @ np.random.randn(nx,N)
        # PF weight update
        log_w = -(g_i(x_pf[:,:,t],u[:,t]) - y[:,t])**2/2/R
        w[t,:] = np.exp(log_w - np.max(log_w))
        w[t,:] = w[t,:]/np.sum(w[t,:])
    
    # Sample trajectory to condition on
    star = systematic_resampling(w[-1,:],1)
    x_prim[:,:,T-1] = x_pf[:,star.astype(int),T-1]
    
    for t in range(T-2,-1,-1):
        star = a[t+1,star.astype(int)]
        x_prim[:,:,t] = x_pf[:,star.astype(int),t]
    
    print('Sampling. k = ',k,'/',K)
    
    # Compute statistics
    
    linear_part = iA@x_prim[:,0,0:T-1] + iB@u[:,0:T-1]

    zeta1 = np.expand_dims((x_prim[0,0,1:T].T - linear_part[0,:]),axis=0)
    z1 = np.expand_dims(phi_1_(x_prim[0,:,0:T-1],u[:,0:T-1]),axis=1)
    zx1 = np.expand_dims(x_prim[0,0,0:T-1],axis=0)
    zu1 = u[:,0:T-1]
    
    zeta2 = np.expand_dims((x_prim[1,0,1:T].T - linear_part[1,:]),axis=0)
    z2 = np.expand_dims(phi_2_(x_prim[:,0,0:T-1],u[:,0:T-1]),axis=1)
    zx2 = x_prim[:,0,0:T-1]
    zu2 = u[:,0:T-1]
    
    # Propose a new jump model
    
    #n1 = np.random.choice(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2]))
    n1 = np.random.geometric(p1)-1
    #pts1 = np.sort(np.array([-L[0][1]*100, (L[0][1]-2*L[0][1], L[0][1]*100]),axis=None)
    pts1 = np.sort(np.hstack([-L[:,1]*100, L[:,1]-2*L[:,1]*np.random.random([1,n1]).flatten(), L[:,1]*100]))
    #pts1 = np.ndarray.sort(np.asanyarray([-L[:,1]*100, L[:,1]-2*L[:,1]*np.random.random(np.array([1,n1])).flatten(), L[:,1]*100],dtype='object'))
    
    # Compute its statistics and marginal likelihood
    
    #zp1 = np.greater_equal(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(pts1[-1][:,None],np.ones((n_basis_1,1))))*np.less(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(pts1[:-1][:,None],np.ones((n_basis_1,1))))@np.tile(phi_1(zx1,zu1),(n1+1,1))
    
    #zp1 = np.multiply(np.tile(zx1,(n_basis_1*(n1+1),1))>= np.multiply((np.tile(zx1,(n_basis_1*(n1+1),1))<np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))).astype(int),np.tile(phi_1_(zx1,zu1),(n1+1,1))))
    ##zp1 = np.multiply(np.greater_equal(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[0:-1],axis=0).T,np.ones((n_basis_1,1)))),np.multiply((np.tile(zx1,(n_basis_1*(n1+1),1))<np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))),np.tile(phi_1_(zx1,zu1),(n1+1,1))))
    zp1 = np.greater_equal(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[0:-1],axis=0).T,np.ones((n_basis_1,1))))*np.less(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1))))*np.tile(phi_1_(zx1,zu1),(n1+1,1))
    #zp1 = np.greater_equal(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[0:-1],axis=0).T,np.ones((n_basis_1,1))))*np.less(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1))))*np.tile(phi_1_(zx1,zu1),(n1+1,1))
    #zp1 = np.greater_equal(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[:-1],axis=0).T,np.ones((n_basis_1,1))))@np.less(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1))))@np.tile(phi_1(zx1,zu1),(n1+1,1))
    #zp1 = np.multiply(np.multiply(np.greater_equal(np.tile(zx1,[n_basis_1*(n1+1),1]),np.kron(pts1[0:n1],np.ones((n_basis_1,1)))),np.less(np.tile(zx1,[n_basis_1*(n1+1),1]),np.kron(pts1[1:n1+1],np.ones((n_basis_1,1))))),np.tile(phi_1(zx1,zu1),[n1+1,1]))
    prop1 = {}
    
    #prop1 = {'Phi', 'Psi', 'Sig', 'V','marginal_likelihood'}
    prop1['Phi'] = zeta1@zeta1.T
    prop1['Psi'] = zeta1@zp1.T
    prop1['Sig'] = zp1@zp1.T
    prop1['V'] = V1(n1)
    prop1['marginal_likelihood'] = compute_marginal_likelihood(prop1['Phi'],prop1['Psi'],prop1['Sig'],prop1['V'],LambdaQ1,lQ1,T-1)
    Mar_lik.append(prop1['marginal_likelihood'])
    prop1['n'] = n1
    prop1['pts'] = pts1
    
    
    if k > 0:
        # Alternatively staying with the current jump model
        n1 = model_state_1[k]['n'] 
        pts1 = model_state_1[k]['pts']

        # Compute its statistics and marginal likelihood
        #zp1 = np.multiply(np.multiply(np.greater_equal(np.tile(zx1,(n_basis_1*(n1+1),1)),np.tile(np.reshape(np.repeat(pts1[:-1],n_basis_1),(n_basis_1*n1,1)),(1,T))),np.less(np.tile(zx1,(n_basis_1*(n1+1),1)),np.tile(np.reshape(np.repeat(pts1[1:n1+1],n_basis_1),(n_basis_1*n1,1)),(1,T)))),np.tile(phi_1(zx1,zu1),(n1+1,1)))
        #curr1.Phi = np.dot(zeta1,zeta1.T)
        #zp1 = np.multiply(np.greater_equal(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[0:-1],axis=0).T,np.ones((n_basis_1,1)))),np.multiply((np.tile(zx1,(n_basis_1*(n1+1),1))<np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))),np.tile(phi_1_(zx1,zu1),(n1+1,1))))
        zp1 = np.greater_equal(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[0:-1],axis=0).T,np.ones((n_basis_1,1))))*np.less(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1))))*np.tile(phi_1_(zx1,zu1),(n1+1,1))
        curr1 = {}
        curr1['Phi'] = zeta1@zeta1.T
        curr1['Psi'] = zeta1@zp1.T
        curr1['Sig'] = zp1@zp1.T
        curr1['V'] = V1(n1)
        curr1['marginal_likelihood'] = compute_marginal_likelihood(curr1['Phi'],curr1['Psi'],curr1['Sig'],curr1['V'],LambdaQ1,lQ1,T-1)
        curr1['n'] = n1 
        curr1['pts'] = pts1
        
    
    dv = np.random.uniform(low=0,high=1)
    
    if (k == 0) :
    #or dv < min(np.exp(prop1['marginal_likelihood'] - curr1['marginal_likelihood']),1):
        
        jmodel = prop1
        accept1 = 1*(jmodel['n']!=model_state_1[k]['n'])
        print('acc1:',accept1)
    elif dv < min(np.exp(prop1['marginal_likelihood'] - curr1['marginal_likelihood']),1):
        jmodel = prop1
        accept1 = 1*(jmodel['n']!=model_state_1[k]['n'])
        print('############ elif acc1:',accept1)
   
    else:
        jmodel = curr1
        accept1 = 0
        print('else acc1:',accept1)
        
        
    model_state_1[k+1]['A'],model_state_1[k+1]['Q'] = gibbs_param( jmodel['Phi'], jmodel['Psi'], jmodel['Sig'], jmodel['V'], LambdaQ1,lQ1,T-1)
    model_state_1[k+1]['n'] = jmodel['n']
    model_state_1[k+1]['pts'] = jmodel['pts']

    if accept1 > 0:
        print('Accept dim 1! New n1 is ', jmodel['n'],'.')
    
    # Fixed discontinutiy point in f_2
    
    zp2 = np.multiply(np.greater_equal(np.tile(zx2[1,:],(n_basis_2*(n2+1),1)),np.kron(np.expand_dims(pts2[0:-1],axis=0).T,np.ones((n_basis_2,1)))),np.multiply((np.tile(zx2[1,:],(n_basis_2*(n2+1),1))<np.kron(np.expand_dims(pts2[1:],axis=0).T,np.ones((n_basis_2,1)))),np.tile(phi_2_(zx2,zu2),(n2+1,1))))
    
    ##zp2 = np.dot(np.greater_equal(np.tile(zx2[1,:],(n_basis_2*(n2+1),1)),np.tile(np.expand_dims(pts2[0:-1],axis=0).T,[(1,n_basis_2]))),np.less(np.tile(zx2[2,:],[n_basis_2*(n2+1),1]),np.tile(np.reshape(pts2[1:(n2+1)],(n2,1)),[1,n_basis_2])))*np.tile(phi_2(zx2,zu2),[n2+1,1])
    #
    #zp2 = np.multiply(np.multiply(np.greater_equal(np.tile(zx2[1,:],[n_basis_2*(n2+1),1]),np.kron(pts2[0:n2],np.ones((n_basis_2,1)))),np.less(np.tile(zx2[1,:],[n_basis_2*(n2+1),1]),np.kron(pts2[1:n2+1],np.ones((n_basis_2,1))))),np.tile(phi_2(zx2,zu2),[n2+1,1]))
    Phi2 = zeta2@zeta2.T
    Psi2 = zeta2@zp2.T
    Sig2 = zp2@zp2.T
        
    model_state_2[k+1]['A'],model_state_2[k+1]['Q'] = gibbs_param( Phi2, Psi2, Sig2, V2(n2), LambdaQ2,lQ2,T-1)
    model_state_2[k+1]['n'] = n2
    model_state_2[k+1]['pts'] = pts2


## Test

# Remove burn-in
burn_in = np.floor(1*K/4).astype(int,casting='unsafe')
Kb = K-burn_in

# Center test data around same working point as training data

u_test= np.expand_dims(dta['uVal'].values,axis=0)
y_test= np.expand_dims(dta['yVal'].values,axis=0)



u_test = normalize(u_test)
y_test = normalize(y_test)
T_test = u_test.shape[1]




Kn = 2
x_test_sim = np.zeros((nx,T_test+1,Kb*Kn))
y_test_sim = np.zeros((T_test,1,Kb*Kn))

for k in range(Kb):
    Qr = np.zeros((nx,nx))
    pts1 = model_state_1[k+burn_in]['pts']
    n1 = model_state_1[k+burn_in]['n']
    Ar1 = model_state_1[k+burn_in]['A']
    Qr[0,0] = model_state_1[k+burn_in]['Q']
    pts2 = model_state_2[k+burn_in]['pts']
    n2 = model_state_2[k+burn_in]['n']
    Ar2 = model_state_2[k+burn_in]['A']
    Qr[1,1] = model_state_2[k+burn_in]['Q']
    
    
    
    def f_r(x,u):
        
        phi_1_tile = np.tile(phi_1_(x[0,:],u[np.zeros((1,x.shape[1])).astype(int)]),(n1+1,1))
        less_1 = np.tile(x[0,:],(n_basis_1*(n1+1),1))<np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))
        greater_1 = np.tile(x[0,:],(n_basis_1*(n1+1),1))>=np.kron(np.expand_dims(pts1[:-1],axis=0).T,np.ones((n_basis_1,1)))
        f_1 = Ar1@np.multiply(greater_1,np.multiply(less_1,phi_1_tile))
        
        phi_2_tile = np.tile(phi_2_(x,u[np.zeros((1,x.shape[1])).astype(int)]),(n2+1,1))
        less_2 = np.tile(x[1,:],(n_basis_2*(n2+1),1))<np.kron(np.expand_dims(pts2[1:],axis=0).T,np.ones((n_basis_2,1)))
        greater_2 = np.tile(x[1,:],(n_basis_2*(n2+1),1))>=np.kron(np.expand_dims(pts2[:-1],axis=0).T,np.ones((n_basis_2,1)))
        f_2 = Ar2@np.multiply(greater_2,np.multiply(less_2,phi_2_tile))
        return iA@x+iB@u[np.zeros((1,x.shape[1])).astype(int)] + np.vstack((f_1,f_2))
    
    #f_r = lambda x,u: iA@x + iB@u[np.zeros((1,x.shape[1])).astype(int)]+np.array([np.squeeze(Ar1@ (np.greater_equal(np.tile(x[0,:],(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[1:-1],axis=0),np.ones((n_basis_1,1)).astype(int)))* np.less_equal(np.tile(x[0,:],(n_basis_1*(n1+1),1)),np.kron(pts1[1:][:,None],np.ones((n_basis_1,1)).astype(int)))*np.tile(phi_1(x[0,:],u[:,np.zeros((1,x.shape[1])).astype(int)]),(n1+1,1)),
    #                                                                    Ar2@np.greater_equal(np.tile(x[1,:],(n_basis_2*(n2+1),1)),np.kron(pts2[1:-1][:None],np.ones((n_basis_2,1)).astype(int)))*np.less_equal(np.tile(x[1,:],(n_basis_2*(n2.astype(int)+1),1)),np.kron(pts2[1:][:,None],np.ones((n_basis_2,1)).astype(int)))*np.tile(phi_2(x,u[:,np.zeros((1,x.shape[1])).astype(int)]),(n2+1,1))])
    #f_r = lambda x,u: iA@x + iB@u[np.zeros((1,x.shape[1])).astype(int)]+np.array([np.squeeze(Ar1@(np.greater_equal(np.tile(x[0,:],(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[:-1],axis=0).T,np.ones((n_basis_1,1)).astype(int)))*np.less(np.tile(x[0,:],(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)).astype(int)))*np.tile(phi_1(x[0,:],u[np.zeros((1,x.shape[1])).astype(int)]),(n1+1,1)))),
                                                                                 # np.squeeze(Ar2@(np.greater_equal(np.tile(x[1,:],(n_basis_2*(n2+1),1)),np.kron(np.expand_dims(pts2[:-1],axis=0).T,np.ones((n_basis_2,1)).astype(int)))* np.less(np.tile(x[1,:],(n_basis_2*(n2+1),1)),np.kron(np.expand_dims(pts2[1:],axis=0).T,np.ones((n_basis_2,1)).astype(int)))*np.tile(phi_2(x,u[np.zeros((1,x.shape[1])).astype(int)]),(n2+1,1))))])
    
    #f_r = lambda x,u: iA@x + iB@u 
 #   np.concatenate((Ar1@((np.tile(x[0,:],[n_basis_1*(n1+1),1]) >= np.kron(pts1[0:n1],np.ones((n_basis_1,1)))).*(np.tile(x[0,:],[n_basis_1*(n1+1),1]) < np.kron(pts1[1:n1+1],np.ones((n_basis_1,1)))).*np.tile(phi_1(x[0,:],u),[n1+1,1])), \
 #    Ar2@((np.tile(x[1,:],[n_basis_2*(n2+1),1]) >= np.kron(pts2[0:n2],np.ones((n_basis_2,1)))).*(np.tile(x[1,:],[n_basis_2*(n2+1),1]) < np.kron(pts2[1:n2+1],np.ones((n_basis_2,1)))).*np.tile(phi_2(x,u),[n2+1,1]))))
    g_r = g_i
    for kn in range(Kn):
        ki = (k-1)*Kn + kn
        for t in range(T_test):
            x_test_sim[0:2,t+1:t+2,ki] = f_r(x_test_sim[0:2,t:t+1,ki],u_test[:,t]) + np.array([np.random.multivariate_normal(np.zeros(nx),Qr)]).T
            y_test_sim[t,0,ki] = g_r(x_test_sim[0:2,t:t+1,ki],0) + np.random.normal(0,R)
    print('Evaluating. k = ' + str(k) + '/' + str(Kb) + '. n1 = ' + str(model_state_1[k+burn_in]['n']) + ', n2 = ' + str(model_state_2[k+burn_in]['n']))

y_test_sim_med = np.median(y_test_sim,2)
y_test_sim_09 = np.quantile(y_test_sim,0.9,2)
y_test_sim_01 = np.quantile(y_test_sim,0.1,2)

# Compare to linear model
x_l = np.array([[0],[0]])
y_sim_l = np.zeros((1,T_test))

for t in range(T_test):
    y_sim_l[:,t] = iC@x_l
    x_l = iA@x_l + iB*u_test[:,t]

ss=np.squeeze(y_test_sim)
mm = np.mean(ss,axis=1)
std = np.std(ss,axis=1)
plt.plot(mm,'g')
plt.plot(y_test[0],'r')
plt.fill_between(range(T_test),(y_test_sim_09.T)[0],(y_test_sim_01.T)[0])
plt.show()

rmse_ss = np.sqrt(np.mean((y_sim_l-y_test)**2))
rmse_sim = np.sqrt(np.mean((y_test_sim_med-y_test.T)**2))

#plt.plot(y_sim_l[0])
#plt.plot(y_test[0])

fig, ax = plt.subplots()
ax.plot(range(T_test),y_test_sim_med,'r',label='Nonlinear')
ax.plot(range(T_test),y_test_sim_09,'r--')
ax.plot(range(T_test),y_test_sim_01,'r--')
ax.plot(range(T_test),y_sim_l[0],'b--',label='Linear')
ax.plot(range(T_test),y_test[:].flatten(),'k',label='True')
ax.legend(loc='upper right')

plt.show()

Ts=4
# colors = np.array([[240,0,0], [255,128,0], [255,200,40], [0,121,64], [64,64,255], [160,0,192], [0,0,0]])/255

# plt.figure(2, figsize=(8,8))
# #gs = matplotlib.gridspec.GridSpec(4,1,height_ratios=[2,2,1,1])

# #plt.subplot(gs[0])
# tv = Ts*np.arange(T_test)
# plt.fill_between(tv,y_test_sim_09+y_ofs,y_test_sim_01+y_ofs,facecolor=np.array([0.6,0.6,0.6]))
# plt.plot(tv,y_test_sim_med+y_ofs,linewidth=1.5,color=colors[5,:])
# plt.plot(tv,y_test+y_ofs,':',linewidth=1.3,color=colors[6,:])

# plt.xlim((0,tv[-1]))
# plt.ylim((1,12))
# plt.ylabel(u'output (V)')
# plt.xlabel(u'$t$ (s)')
# plt.setp(gca(),xticklabels=[])












   
    #f_1 = lambda x,u: Ai1.dot((np.tile(x[0,:],(n_basis_1*(n1+1),1))>=np.kron(np.expand_dims(pts1[:-1],axis=0).T,np.ones((n_basis_1,1)))).dot((np.tile(x[0,:],(n_basis_1*(n1+1),1))<np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))).dot(np.tile(phi_1(x[0,:],u[np.zeros((1,x.shape[1])).astype(int)]),(n1+1,1)).T).T).T)
    #f_2 = Ai2.dot((np.tile(x[1,:],(n_basis_2*(n2+1),1))>=np.kron(np.expand_dims(pts2[:-1],axis=0).T,np.ones((n_basis_2,1)))).T).dot(np.tile(x[1,:],(n_basis_2*(n2+1),1))<np.kron(np.expand_dimspts2[1:],axis=0).T,np.ones((n_basis_1,1)))).dot(np.tile(phi_2(x,u[np.zeros((1,x.shape[1])).astype(int)]),(n2+1,1)).T).T).T)
    # def f_i(x,u):
        
        #f_1 = Ai1.dot(np.tile(x[0,:],(n_basis_1*(n1+1),1))>=np.kron(pts1[:-1][:,None],np.ones((n_basis_1,1))).T).dot(np.tile(x[0,:],(n_basis_1*(n1+1),1))<np.kron(pts1[1:][:,None],np.ones((n_basis_1,1)))).dot(phi_1(x[0,:],u[np.zeros((1,x.shape[1])).astype(int)]).T).T
        
        #f_2 = Ai2.dot(np.tile(x[1,:],(n_basis_2*(n2+1),1))>=np.kron(pts2[:-1][:,None],np.ones((n_basis_2,1))).T).dot(np.tile(x[1,:],(n_basis_2*(n2+1),1))<np.kron(pts2[1:][:,None],np.ones((n_basis_1,1)))).dot(phi_2(x,u[np.zeros((1,x.shape[1])).astype(int)]).T).T
        
        #return iA.dot(x)+iB.dot(u[:,np.zeros((1,x.shape[1])).astype(int)] + np.concatenate((f_1,f_2),axis=0)
 
    
   # def f_i(x,u):
        
        #return iA@x + iB@u + np.array([Ai1 * ((x[0]>=pts1[0:-1].reshape(1,-1)).T *(x[0]<pts1[1:-1].reshape(1,-1)).T *phi_1(x[0,:],u[np.zeros((1,x.shape[1])).astype(int)])).T, Ai2 * ((x[1]>=pts2[0:-1].reshape(1,-1)).T *(x[1]<pts2[1:-1].reshape(1,-1)).T *phi_2(x[:,:],u[np.zeros((1,x.shape[1])).astype(int)])).T])

    #f_i = lambda x,u: iA.dot(x) + iB.dot(u[np.zeros((1,x.shape[1])).astype(int)])+
                #np.array([np.squeeze(Ai1.dot(np.greater_equal(np.dot(np.tile(x[0,:],(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[:-1],axis=0).T,np.ones((n_basis_1,1)).astype(int))),np.dot(np.less(np.dot(np.tile(x[0,:],(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)).astype(int))),(np.tile(phi_1(x[0,:],u[np.zeros((1,x.shape[1])).astype(int)]),(n1+1,1)))))))),
                             #np.squeeze(Ai2@(np.greater_equal(np.tile(x[1,:],(n_basis_2*(n2+1),1)),np.kron(np.expand_dims(pts2[:-1],axis=0).T,np.ones((n_basis_2,1)).astype(int))), np.dot(np.less(np.dot(np.tile(x[1,:],(n_basis_2*(n2+1),1)),np.kron(np.expand_dims(pts2[1:],axis=0).T,np.ones((n_basis_2,1)).astype(int))),(np.tile(phi_2(x,u[np.zeros((1,x.shape[1])).astype(int)]),(n2+1,1)))]))))
    
    
