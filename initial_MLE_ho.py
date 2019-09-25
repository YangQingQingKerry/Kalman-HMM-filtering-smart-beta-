#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

class MLE:
    def __init__(self,state, tns): #training sample
        self.dir_path="./data/pca/keyFactor"
        self.df=pd.read_excel(self.dir_path+os.sep+'keyFactor_2009-09-11.xlsx')
        self.df.set_index('Date', inplace=True)
        self.df=(self.df.shift(-21)/self.df-1).dropna()
        self.f=(self.df-self.df.min())/(self.df.max()-self.df.min())
        self.min=self.df.min().values
        self.max=self.df.max().values
        self.k=len(self.f.columns)
        self.state=state
        self.tns=tns

    def phi_k(self, t, s,  args):
        from numpy.linalg import inv, det
        from math import exp, sqrt, pi

        v1=self.f.iloc[t].values
        v0= np.array([0.]*self.k) if t==0 else self.f.iloc[t-1].values
        v=v1-args[s*self.k:(s+1)*self.k]*v0-args[(s+self.state)*self.k:(s+1+self.state)*self.k]
        Sig=np.diag(args[(s+2*self.state)*self.k:(s+1+2*self.state)*self.k])
        D=inv(Sig)
        return exp(-0.5*np.sum(v*D*v.reshape(self.k,1)))/sqrt((2*pi)**self.k*det(Sig))


    def obj(self, args):
        '''
        args(e.g., state=2):
        =====
             p_ij: args[:16]=(pi_11, pi_12, pi_13,...)
             kappa_i: args[16: 16+2*self.k]
             gamma_i: args[16+2*self.k:16+4*self.k]
             sigma_i^2: args[16+4*self.k:]
        '''
        from math import log
        p_o=[1/self.state**2]*self.state**2
        f_t=[p_o[i]*self.phi_k(0,i, args[self.state**4:]) for i in range(self.state) for j in range(self.state)]
        ft=sum(f_t)
        if ft==0: return np.nan  #stopError
        p=np.array(f_t)/ft
        llh=-log(ft)
        for t in range(1, self.tns):
            f_t=[]
            for _r in range(self.state):
                for _s in range(self.state):
                    for _t in range(self.state):
                        #i->j
                        i=_s*self.state+_t
                        j=_r*self.state+_s
                        f_t.append(p[i]*args[j*self.state+i]*self.phi_k(t, _r, args[self.state**4:]))
            ft=sum(f_t)
            if ft==0: return np.nan #stopError
            f_t=np.array(f_t).reshape(self.state**2, self.state)
            p=f_t.sum(axis=1)/ft
            llh-=log(ft)
        pd.DataFrame(args).to_excel('./output/init_mle/arg_{}state_ho.xlsx'.format(self.state))
        return llh

    def con1(self, args):
        return args[0]+args[8]-1.
    def con2(self, args):
        return args[1]+args[9]-1.
    def con3(self, args):
        return args[2]
    def con4(self, args):
        return args[3]
    def con5(self, args):
        return args[4]
    def con6(self, args):
        return args[5]
    def con7(self, args):
        return args[6]+args[14]-1.
    def con8(self, args):
        return args[7]+args[15]-1.
    def con9(self, args):
        return args[10]
    def con10(self, args):
        return args[11]
    def con11(self, args):
        return args[12]
    def con12(self,args):
        return args[13]


    def optimz(self, args):
        from scipy.optimize import minimize
        b1=(0.,  1)
        b2=(-100, 100) #from -100% to 100%
        b3=(0.0001, 1)
        bnds=tuple([b1]*(self.state**4) +[b3]*(self.k*self.state)+[b2]*(self.k*self.state)+[b3]*(self.k*self.state))
        cons=({'type':'eq', 'fun': self.con1},
              {'type':'eq', 'fun': self.con2},
              {'type':'eq', 'fun': self.con3},
              {'type':'eq', 'fun': self.con4},
              {'type':'eq', 'fun': self.con5},
              {'type':'eq', 'fun': self.con6},
              {'type':'eq', 'fun': self.con7},
              {'type':'eq', 'fun': self.con8},
              {'type':'eq', 'fun': self.con9},
              {'type':'eq', 'fun': self.con10},
              {'type':'eq', 'fun': self.con11},
              {'type':'eq', 'fun': self.con12})
        sol=minimize(self.obj, args, method='SLSQP', bounds=bnds, constraints=cons)
        return sol




if __name__=="__main__":
    state=2
    tns=63
    ins=MLE(state, tns)




    df=ins.f.iloc[:tns]
    df_1=df.shift(1).fillna(0)
    TT1=(df*df_1).sum()
    T=df.sum()
    T1=df_1.sum()
    T1_sq=(df_1**2).sum()
    theta=(tns*TT1-T*T1)/(tns*T1_sq-T1**2)
    mu=(T-T1*theta)/tns
    rm=df-theta*df_1-mu
    sigma2=(rm**2).sum()/tns
    args=[0.5,0.5,0,0,
          0,0,0.5,0.5,
          0.5,0.5,0,0,
          0,0,0.5,0.5]+list(theta)*2+list(mu)\
       +list(mu+np.random.normal(0,0.01, ins.k))+list(sigma2)*2
    args=np.array(args)


    #args=pd.read_excel('./output/init_mle/arg_2state_ho.xlsx')[0].values+\
    #np.array([0.]*16+list(np.random.normal(0,0.0001,30)))



    print(ins.obj(args))
    print(ins.optimz(args))













