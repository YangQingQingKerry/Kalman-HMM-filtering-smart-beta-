#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

class MLE:
    def __init__(self,state, tns): #training sample
        self.dir_path="./data/pca/keyFactor"
        self.df=pd.read_excel(self.dir_path+os.sep+'keyFactor_2009-09-11.xlsx')
        #self.df=self.df[['Date','factor1']]
        self.df.set_index('Date', inplace=True)
        self.f=(self.df-self.df.min())/(self.df.max()-self.df.min())
        self.min=self.df.min().values
        self.max=self.df.max().values
        self.k=len(self.f.columns)
        self.state=state
        self.tns=tns

    def phi_k(self, t, s,  args):
        from math import exp, sqrt, pi

        phi=1
        v1=self.f.iloc[t].values
        v0= np.array([0.]*self.k) if t==0 else self.f.iloc[t-1].values
        for i in range(self.k):
            phi*=(exp(-0.5*(v1[i]-args[i+s*self.k]*v0[i]-args[i+(s+self.state)*self.k])**2\
                      /args[i+(2*self.state+s)*self.k])/sqrt(2*pi*args[i+(2*self.state+s)*self.k]))
        return phi

    def obj(self, args):
        '''
        args(e.g., state=2):
        =====
             p_ij: args[:4]=(pi_11, pi_12, pi_21, pi_22)
             kappa_i: args[4: 4+2*self.k]
             gamma_i: args[4+2*self.k:4+4*self.k]
             sigma_i^2: args[4+4*self.k:]
        '''
        from math import log
        p_o=[1/self.state]*self.state
        f_t=[]
        for i in range(self.state):
            f_t.append(p_o[i]*self.phi_k(0,i, args[self.state**2:]))
        ft=sum(f_t)
        if ft==0: return np.nan  #stopError
        p=np.array(f_t)/ft
        llh=-log(ft)
        for t in range(1, self.tns):
            f_t=[]
            for i in range(self.state):
                for j in range(self.state):
                    f_t.append(p[i]*args[i*self.state+j]*self.phi_k(t, j, args[self.state**2:]))
            ft=sum(f_t)
            if ft==0: return np.nan #stopError
            f_t=np.array(f_t).reshape(self.state, self.state)
            p=f_t.sum(axis=0)/ft
            llh-=log(ft)
        pd.DataFrame(args).to_excel('./output/init_mle/arg_{}state.xlsx'.format(self.state))
        return llh


    def optimz(self, args):
        from scipy.optimize import minimize
        b1=(0.,  1)
        b2=(-100, 100) #from -100% to 100%
        b3=(0.0001, 100)
        bnds=tuple([b1]*(self.state**2) +[b1]*(self.k*self.state)+[b2]*(self.k*self.state)+[b3]*(self.k*self.state))
        cons=tuple([{'type':'eq', 'fun':lambda x: sum(x[i*self.state:(i+1)*self.state])-1} \
                     for i in range(self.state)])
        sol=minimize(self.obj, args, method='SLSQP', bounds=bnds, constraints=cons)
        return sol




if __name__=="__main__":
    state=1
    args=[1/state]*(state**2)
    tns=63
    ins=MLE(state, tns)

    '''
    1-state model parameter estimates'
    '''
    df=ins.f.iloc[:tns]
    df_1=df.shift(1).fillna(0)
    Tt=(df*df_1).sum()
    T=df.sum()
    t=df_1.sum()
    t2=(df_1**2).sum()
    alpha=np.array(list(map(lambda x: max(0.0001,x), (tns*Tt-T*t)/(tns*t2-t**2))))
    theta=(Tt-alpha*t2)/t
    rm=df-alpha*df_1-theta
    eta_2=(rm**2).sum()/tns



    kappa=list(alpha)*state
    gamma=list(theta.values)*state
    sigma=list(eta_2.values)*state  #add some noise to 1-state model parameter estimates
    args=args+ kappa+ gamma+sigma
    pd.DataFrame(args).to_excel('./output/init_mle/arg_1state.xlsx')


    print(ins.obj(args))
    print(ins.optimz(args))













