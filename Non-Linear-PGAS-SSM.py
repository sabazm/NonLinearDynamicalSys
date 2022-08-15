#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:52:19 2022

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
import scipy.special as sp
import scipy.stats as stats
import scipy.linalg as linalg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.plotting import register_matplotlib_converters
from scipy.stats import norm
import scipy.stats as ss
from nfoursid.kalman import Kalman
from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace




register_matplotlib_converters()
np.seterr(divide='ignore', invalid='ignore')
sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
plt.rcParams['figure.figsize'] = [10, 4]
plt.rcParams['figure.dpi'] = 100


np.random.seed(1)

def gibbs_param( Phi, Psi, Sigma, V, Lambda, l, T ):


    nb = V.shape[0]
    nx = np.size(Phi)
    
    M = np.zeros((nx,nb))
    Phibar = Phi + ((M @ np.linalg.inv(V))@M.T)
    Psibar = Psi + (M @ np.linalg.inv(V))
    Sigbar = Sigma + np.linalg.inv(V)
    cov_M = Lambda+Phibar-((Psibar @ np.linalg.inv(Sigbar)) @ Psibar.T)
    cov_M_sym = 0.5*(cov_M + cov_M.T) 
    Q = ss.invwishart.rvs(scale=cov_M_sym,df=T+l)
    X = (np.random.normal(loc=0,scale=0.5,size=(nx,nb)))
    post_mean = Psibar @ np.linalg.inv(Sigbar)
    A = post_mean + cholesky(Q)@ X @ cholesky(np.linalg.inv(Sigbar))
    return A, Q

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
 
def normalize(variable):
        normalized_var = (variable - np.mean(variable))
        return normalized_var


dt = pd.read_csv('FLX_FR-Pue_FLUXNET2015_SUBSET_HH_2000-2014_2-4.csv')

dt[['TIMESTAMP_START']] = (dt[['TIMESTAMP_START']].applymap(str).applymap(
    lambda s: "{}/{}/{} {}:{}".format(s[0:4], s[4:6], s[6:8], s[8:10], s[10:12])))


cols = ['TIMESTAMP_START','TA_F','NIGHT','NEE_VUT_50','RECO_NT_VUT_50']

dt['TIMESTAMP_START'] = dt['TIMESTAMP_START'].astype('datetime64[ns]')

#dt.set_index(dt['TIMESTAMP_START'], inplace=True)
mask = (dt['TIMESTAMP_START'] >= '2006/07/01 00:00') & (dt['TIMESTAMP_START'] < '2006/08/01 00:00')
ts1 = dt.loc[mask]
dta = ts1.loc[:,cols]
dta.reset_index(drop=True)
#dta.set_index(dta['TIMESTAMP_START'], inplace=True)
tresh = 0.8
tr = int(dta.shape[0]*tresh)
tr = dta.index[0]+tr
dta_copy = ts1.copy()
#index_day = np.where(dta['NIGHT']!=1)
train =dta.loc[:tr]
test = dta.loc[tr-5:]
df_train = train.loc[(train["NIGHT"]== 1), cols]
df_test = test.loc[(test["NIGHT"] != 1), cols]
#dta['TA_F'] = (np.where(dta['NIGHT']!=1,np.nan,dta['TA_F']))
#dta['NEE_VUT_50'] = (np.where(dta['NIGHT']!=1,np.nan,dta['NEE_VUT_50']))

#u= np.expand_dims(df_train['TA_F'].values,axis=0)
#y= np.expand_dims(df_train['NEE_VUT_50'].values,axis=0)


u= np.expand_dims(df_train['TA_F'].values,axis=0)
y= np.expand_dims(df_train['NEE_VUT_50'].values,axis=0)

u_ofs = np.mean(u)
y_ofs = np.mean(y)

T = u.shape[1]

#u = u[:,0:T] - u_ofs
#y = y[:,0:T] - y_ofs

T = u.shape[1]

#u_mean = u.mean()
#y_mean = y.mean()
#u = normalize(u)
#y = normalize(y)
u_test = np.expand_dims(df_test['TA_F'].values,axis=0)
y_test = np.expand_dims(df_test['NEE_VUT_50'].values,axis=0)
T_test = u_test.shape[1]
#u_test = u_test - u_ofs
#y_test = y_test - y_ofs

###################################################
#Linear Model
###################################################

NUM_TRAINING_DATAPOINTS = u.shape[1] # create a training-set by simulating a state-space model with this many datapoints
NUM_TEST_DATAPOINTS = u_test.shape[1] # same for the test-set
INPUT_DIM = 1
OUTPUT_DIM = 1
INTERNAL_STATE_DIM = 1  # actual order of the state-space model in the training- and test-set
NOISE_AMPLITUDE = 0.1  # add noise to the training- and test-set
FIGSIZE = 8
figsize = (1.3 * FIGSIZE, FIGSIZE)

state_space = pd.DataFrame(df_train.loc[:,['TA_F','NEE_VUT_50']])
nfoursid = NFourSID(
    state_space,  # the state-space model can summarize inputs and outputs as a dataframe
    output_columns=['NEE_VUT_50'],
    input_columns=['TA_F'],
    num_block_rows=10
)
nfoursid.subspace_identification()

ORDER_OF_MODEL_TO_FIT = 2

state_space_identified, covariance_matrix = nfoursid.system_identification(rank=ORDER_OF_MODEL_TO_FIT)
iA = state_space_identified.a
iB = state_space_identified.b
iC = state_space_identified.c
iB = iB*iC[0,1]
iC[0,1] = 1.
#print(state_space_identified.d)
kalman = Kalman(state_space_identified, covariance_matrix)

#state_space = StateSpace.(A, B, C, D)  # new data for the test-set
for i in range(NUM_TEST_DATAPOINTS-1):

    kalman.step(  np.expand_dims(y_test[:,i],axis=0), np.expand_dims(u_test[:,i],axis=0))
 
b = kalman.to_dataframe()

fig = plt.figure(figsize=figsize)
kalman.plot_filtered(fig)
kalman.plot_predicted(fig)
fig.tight_layout()
plt.show()

def g_i(x,u=None):
    return np.array([0,1])@x

R = 0.25
nx = 2 
nu = 1 
ny = 1


# Parameters for the algorithm, priors, and basis functions
K = 5000
N = 50

# Basis functions for f:
n_basis_u = 7
n_basis_x1 = 7
n_basis_x2 = 7

#L = np.zeros((1,1,3))

L = np.expand_dims([8., 15.,9.],axis=0)


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


S_SE = lambda w, ell : np.sqrt(2*np.pi*ell**2)*np.exp(-(np.square(w)/2)*ell**2)
def V1(n1):
    return 100*np.diag(np.repeat(np.prod(S_SE(np.sqrt(lambda_1),np.tile([3,3],[n_basis_1,1])),1),n1+1))

def V2(n2):
    return 100*np.diag(np.repeat(np.prod(S_SE(np.sqrt(lambda_2),np.tile([3,3,3],[n_basis_2,1])),1),n2+1))


#V1 = lambda n1: 100*np.diag(np.squeeze(np.tile(np.prod(S_SE(np.sqrt(lambda_1),np.tile(np.array([3,3]),(n_basis_1,1))),keepdims=True,axis=1),(n1+1,1))))
#V2 = lambda n2: 100*np.diag(np.squeeze(np.tile(np.prod(S_SE(np.sqrt(lambda_2),np.tile(np.array([3,3,3]),(n_basis_2,1))),keepdims=True,axis=1),(n2+1,1))))
phi_1_ = lambda x,u:np.prod(np.multiply(L**(-1/2),np.sin(np.multiply(np.multiply((np.pi*jv_1),((np.transpose(np.vstack((u,x))[:,:,None], (2,1,0)))+L[np.zeros((x.shape[1])).astype(int),:])),1.0/2*L))),2)
phi_2_ = lambda x,u:np.prod(np.multiply(L**(-1/2),np.sin(np.multiply(np.multiply((np.pi*jv_2),((np.transpose(np.vstack((u,x))[:,:,None], (2,1,0)))+L[np.zeros((x.shape[1])).astype(int),:])),1.0/2*L))),2)
# Priors for Q
lQ1 = 1000
lQ2 = 1000
LambdaQ1 = 1.0*np.eye(1)
LambdaQ2 = 1.0*np.eye(1)
# model_state is the posterior on the parameters of the nonlinear model.
model_state_1 = [{} for i in range(K+2)]
model_state_2 = [{} for i in range(K+2)]

model_state_1[0]['A'] = np.zeros((1,n_basis_1))
model_state_1[0]['Q'] = 1
model_state_1[0]['n'] = 0
model_state_1[0]['pts'] = np.array([-L[0][1],L[0][1]])

model_state_2[0]['A'] = np.zeros((1,2*n_basis_2))
model_state_2[0]['Q'] = 1
model_state_2[0]['n'] = 1
model_state_2[0]['pts'] = np.array([-L[0][2],2.4,L[0][2]])
####################

####################
def f_i(x,u):
    f_1 = (iA[0:1,:]@x)+ (iB[0:1]@u) + (model_state_1p[0]@phi_1(x[0:1,:],u))
    f_2 = (iA[1:2,:]@x) + (iB[1:2]@u) + (model_state_2p[0]@phi_2(x,u))
    return np.vstack((f_1,f_2))
#f_i = lambda x,u: np.array([np.dot(iA[0,:],x)+ np.dot(iB[0,:],u) + np.squeeze(np.dot(model_state_1p[0],phi_1(x[0,:],u))), np.dot(iA[1,:],x) + np.dot(iB[1,:],u) + np.squeeze(np.dot(model_state_2p[0],phi_2(x,u)))])

ys = np.zeros((5,T))
for i in range(5):

    model_state_1p = gibbs_param(0, 0, 0, V1(0), LambdaQ1,lQ1,0)
    model_state_2p = gibbs_param(0, 0, 0, V2(0), LambdaQ2,lQ2,0)

    xs = np.zeros((2,1))
    

    for t in range(T):
        ys[i,t]=(g_i(xs,0))
        xs = f_i(xs,u[:,t])

    plt.plot(ys[i,:])

    #plt.draw()
    plt.pause(0.01)

plt.show()





##############################

##############################

#phi_1_ = lambda x1,u: np.prod(np.multiply(L[:,:]**(-1/2),np.sin(np.multiply(np.multiply((np.pi*jv_1),((np.transpose(np.vstack((u,x1))[:,:,None], (2,1,0)))+L[np.zeros((x1.shape[0])).astype(int),:])),1.0/2*L[:,:]))),2)
#def phi_1_(x1,u):
#    return np.prod(L[0,0:2]**(-1/2)*np.sin(np.pi*jv_1*(np.vstack((u,x1)) + L[0,:2])/(2*L[0,:2])),2)

p1 = 0.99


# Pre-allocate
x_prim = np.zeros((nx,1,T))

############################
#MCMC Algorithm
############################
## Run learning algorithm


#The code is running a particle filter with ancestor sampling (CPF-AS) to sample a trajectory from the posterior distribution of the state given the measurements. It then uses this sample to update the parameters of the model. 
#The model is a simple linear system with a discontinuity in the state transition function. The discontinuity is modelled using a mixture of Gaussians. The mixture is parametrized by the number of components, the means and the variances of each component. The number of components is updated using a Gibbs sampler. The means and variances are updated using a Gibbs sampler. 

Mar_lik=[]
zeta = []

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
    
    
    
    # def f_i(x,u):
        
    #     phi_1_tile = np.tile(phi_1_(x,u[np.zeros((1,x.shape[1])).astype(int)]),(n1+1,1))
    #     less_1 = np.tile(x,(n_basis_1*(n1+1),1))<np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))
    #     greater_1 = np.tile(x,(n_basis_1*(n1+1),1))>=np.kron(np.expand_dims(pts1[:-1],axis=0).T,np.ones((n_basis_1,1)))
    #     f_1 = Ai1 @ np.multiply(greater_1,np.multiply(less_1,phi_1_tile))
    #     return iA@x+iB@u[np.zeros((1,x.shape[1])).astype(int)] + f_1

    def f_i(x,u):
        
        phi_1_tile = np.tile(phi_1(x[0],u[np.zeros((1,x.shape[1])).astype(int)]),(n1+1,1))
        less_1 = np.tile(x[0],(n_basis_1*(n1+1),1))<np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))
        greater_1 = np.tile(x[0],(n_basis_1*(n1+1),1))>=np.kron(np.expand_dims(pts1[:-1],axis=0).T,np.ones((n_basis_1,1)))
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
        log_w = -(g_i(x_pf[:,:,t],u[:,t])- y[:,t])**2/2/R
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
    
    #plt.plot(linear_part[0])
    #plt.plot(y[0])
    #plt.show()

    # z is the basis functions for the nonlinear part of the model.
    zeta1 = np.expand_dims((x_prim[0,0,1:T].T - linear_part[0,:]),axis=0)
    zeta.append(zeta1)
    z1 = np.expand_dims(phi_1(x_prim[0,:,0:T-1],u[:,0:T-1]),axis=1)
    zx1 = np.expand_dims(x_prim[0,0,0:T-1],axis=0)
    zu1 = u[:,0:T-1]
    
    zeta2 = np.expand_dims((x_prim[1,0,1:T].T - linear_part[1,:]),axis=0)
    z2 = np.expand_dims(phi_2(x_prim[:,0,0:T-1],u[:,0:T-1]),axis=1)
    zx2 = x_prim[:,0,0:T-1]
    zu2 = u[:,0:T-1]
    # Propose a new jump model
    
    #n1 = np.random.choice(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2]))
    n1 = np.random.geometric(p1)-1
    #n1 = 0
    #pts1 = np.sort(np.array([-L[0][1]*100, (L[0][1]-2*L[0][1], L[0][1]*100]),axis=None)
    pts1 = np.sort(np.hstack([-L[:,1]*100, L[:,1]-2*L[:,1]*np.random.random([1,n1]).flatten(), L[:,1]*100]))
    #pts1 = np.ndarray.sort(np.asanyarray([-L[:,1]*100, L[:,1]-2*L[:,1]*np.random.random(np.array([1,n1])).flatten(), L[:,1]*100],dtype='object'))
    
    # Compute its statistics and marginal likelihood
    
    
    zp1 = np.greater_equal(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[0:-1],axis=0).T,np.ones((n_basis_1,1))))*np.less(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1))))*np.tile(phi_1(zx1,zu1),(n1+1,1))
   
    prop1 = {}
    
    #prop1 = {'Phi', 'Psi', 'Sig', 'V','marginal_likelihood'}
    
    # Phi = zeta*zeta' is the matrix of samples from the nonlinear model.
    # Psi = zeta*z' is the matrix of samples from the linear model.
    # Sig = z*z' is the matrix of basis functions.
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
        zp1 = np.multiply(np.greater_equal(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[0:-1],axis=0).T,np.ones((n_basis_1,1)))),np.multiply(np.less(np.tile(zx1,(n_basis_1*(n1+1),1)),np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))),(np.tile(phi_1(zx1,zu1),(n1+1,1)))))
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
        print('Marg:',min(np.exp(prop1['marginal_likelihood'] - curr1['marginal_likelihood']),1) )
   
    else:
        jmodel = curr1
        accept1 = 0
        print('else acc1:',accept1)
        
        
    model_state_1[k+1]['A'],model_state_1[k+1]['Q'] = gibbs_param( jmodel['Phi'], jmodel['Psi'], jmodel['Sig'], jmodel['V'], LambdaQ1,lQ1,T-1)
    model_state_1[k+1]['n'] = jmodel['n']
    model_state_1[k+1]['pts'] = jmodel['pts']

    if accept1 > 0:
        print('Accept dim 1! New n1 is ', jmodel['n'],'.')
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


    
#####################################
## Plotting
#####################################

# length = 200
# for k in range(K):

#     if k == 0:
#         plt.figure(figsize=(10, 6))
#         plt.plot(x_prim[0,0,0:T], color='g', label='True simulated trajectory')
#         plt.plot(x_pf[0,np.random.choice(np.arange(0,N),length,False),:].T, color='m', label='PF trajectory')
#         plt.legend(loc='best')
#         plt.show(block=False)
#     if ((k>0) & (k%100==0)):
#         plt.figure(figsize=(10, 6))
#         plt.plot(x_prim[0,0,0:T], color='g', label='True simulated trajectory')
#         plt.plot(x_pf[0,np.random.choice(np.arange(0,N),length,False),:].T, color='m', label='PF trajectory')
#         plt.legend(loc='best')
#         plt.show(block=False)

# # Plot marginal likelihood
# plt.figure(figsize=(10, 4.7))
# plt.scatter(np.arange(K+1),Mar_lik,marker='x',color='k',label='Marginal likelihood')
# plt.show(block=False)

# # Plot parameter evolution
# colors = ['r','g','b']
# plt.figure(figsize=(10, 6))
# for i in range(n_basis_u):
#     plt.subplot(nx,1,i+1)
#     for j in range(n_basis_u):
#         plt.plot(np.squeeze(np.asarray([model_state_1[k]['A'][i,j] for k in range(0,K+1)])),color=colors[j%len(colors)])
#     plt.legend(loc='best')
# plt.show(block=False)

# plt.show()


## Test

# Remove burn-in
burn_in = min(int(K/4),2000)
Kb = K-burn_in

# Center test data around same working point as training data


#index_night = np.where(dta_copy['NIGHT']==1)
#df_test = dta_copy.loc[dta_copy, cols]

# u_test= np.expand_dims(dta_copy['TA_F'].values,axis=0)
# y_test= np.expand_dims(dta_copy['NEE_VUT_50'].values,axis=0)


# u_test = normalize(u_test)
# y_test = normalize(y_test)
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
    
    #f_r = lambda x,u: iA.dot(x) + iB.dot(u) + np.vstack([Ar1.dot(np.prod(L**(-1/2)*np.sin(np.pi*jv_1*(np.vstack((u,x)) + L)/(2*L)),0))])
    
    # def f_r(x,u):
        
    #     phi_1_tile = np.tile(phi_1(x,u[np.zeros((1,x.shape[1])).astype(int)]),(n1+1,1))
    #     less_1 = np.tile(x,(n_basis_1*(n1+1),1))<np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))
    #     greater_1 = np.tile(x,(n_basis_1*(n1+1),1))>=np.kron(np.expand_dims(pts1[:-1],axis=0).T,np.ones((n_basis_1,1)))
    #     f_1 = Ar1@np.multiply(greater_1,np.multiply(less_1,phi_1_tile))
        
    #     return iA@x+iB@u[np.zeros((1,x.shape[1])).astype(int)] + f_1
 
    def f_r(x,u):
        
        phi_1_tile = np.tile(phi_1(x[0,:],u[np.zeros((1,x.shape[1])).astype(int)]),(n1+1,1))
        less_1 = np.tile(x[0,:],(n_basis_1*(n1+1),1))<np.kron(np.expand_dims(pts1[1:],axis=0).T,np.ones((n_basis_1,1)))
        greater_1 = np.tile(x[0,:],(n_basis_1*(n1+1),1))>=np.kron(np.expand_dims(pts1[:-1],axis=0).T,np.ones((n_basis_1,1)))
        f_1 = Ar1@np.multiply(greater_1,np.multiply(less_1,phi_1_tile))
        
        phi_2_tile = np.tile(phi_2(x,u[np.zeros((1,x.shape[1])).astype(int)]),(n2+1,1))
        less_2 = np.tile(x[1,:],(n_basis_2*(n2+1),1))<np.kron(np.expand_dims(pts2[1:],axis=0).T,np.ones((n_basis_2,1)))
        greater_2 = np.tile(x[1,:],(n_basis_2*(n2+1),1))>=np.kron(np.expand_dims(pts2[:-1],axis=0).T,np.ones((n_basis_2,1)))
        f_2 = Ar2@np.multiply(greater_2,np.multiply(less_2,phi_2_tile))
        return iA@x+iB@u[np.zeros((1,x.shape[1])).astype(int)] + np.vstack((f_1,f_2))
    g_r = g_i
    for kn in range(Kn):
        ki = (k-1)*Kn + kn
        for t in range(T_test):
            x_test_sim[0:2,t+1:t+2,ki] = f_r(x_test_sim[0:2,t:t+1,ki],u_test[:,t]) + np.array([np.random.multivariate_normal(np.zeros(nx),Qr)]).T
            y_test_sim[t,0,ki] = x_test_sim[0,t:t+1,ki] + np.random.normal(0,R)
    print('Evaluating. k = ' + str(k) + '/' + str(Kb) + '. n1 = ' + str(model_state_1[k+burn_in]['n'])+ '. n2 = ' + str(model_state_2[k+burn_in]['n']))

y_test_sim_med = np.mean(y_test_sim,2)
y_test_sim_std = np.std(y_test_sim,2)
y_test_sim_09 = np.quantile(y_test_sim,0.9,2)
y_test_sim_01 = np.quantile(y_test_sim,0.1,2)

rmse_sim = np.sqrt(np.mean((y_test_sim_med-y_test.T)**2))

# Compare to linear model
x_l = np.array([[0.],[0.]])
y_sim_l = np.zeros((1,T_test))
iC[0,0] = 0.0
for t in range(T_test):
    y_sim_l[0,t] = iC @ x_l
    x_l = iA @x_l + iB @u_test[:,t]
    #print(y_sim_l)
    #print(x_l)

#plt.plot(y_sim_l[0])
#plt.plot(y_test[0],'r')
plt.plot(y_test_sim_med+2.2,'g')
plt.plot(y_test[0]+2.2,'r')
#plt.plot(dta_copy['RECO_NT_VUT_50'])
plt.fill_between(range(T_test),(y_test_sim_09.T)[0],(y_test_sim_01.T)[0])
plt.show()




plt.plot(x_prim[0,0,:])
plt.plot(u[0])






