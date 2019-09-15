#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os

class MLE:
    def __init__(self,
                 tns): #training sample
        self.dir_path="./data/pca/keyFactor"
        self.df=pd.read_excel(self.dir_path+os.sep+'keyFactor_2009-01-19.xlsx')
        self.df.set_index('Date', inplace=True)
        self.r=self.df/self.df.shift(1)-1
        self.r.iloc[0]=self.df.iloc[0].values-1
        self.r=100*self.r #0.1%
        self.k=len(self.r.columns)
        self.tns=tns




    def phi_k(self, t, s,  args):
        from math import exp, sqrt, pi
        import numpy as np

        phi=1
        v1=self.r.iloc[t].values
        v0=np.array([0.]*self.k) if t==0 else self.r.iloc[t-1].values
        for i in range(self.k):
            phi*=(exp(-0.5*(v1[i]-args[i+s*self.k]*v0[i]-args[i+(s+2)*self.k])**2/args[i+(4+s)*self.k])/sqrt(2*pi*args[i+(4+s)*self.k]))
        return phi





    def obj(self, args):
        '''
        args:
        =====
             p_ij: args[:4]=(pi_11, pi_12, pi_21, pi_22)
             mu_i: args[4: 4+2*self.k]
             sigma_i^2: args[4+2*self.k:]

        '''
        from math import log
        import numpy as np



        f_t1=args[2]/(1-args[0]+args[2])*self.phi_k(0, 0, args[4:])
        f_t2=args[1]/(1-args[3]+args[1])*self.phi_k(0, 1, args[4:])
        f_t=f_t1+f_t2
        if f_t==0: return np.nan  #stopError
        p1=f_t1/f_t
        p2=f_t2/f_t
        llh=-log(f_t)
        for t in range(1, self.tns):
            f_t11=p1*args[0]*self.phi_k(t, 0, args[4:])
            f_t12=p2*args[2]*self.phi_k(t, 0, args[4:])
            f_t21=p1*args[1]*self.phi_k(t, 1, args[4:])
            f_t22=p2*args[3]*self.phi_k(t, 1, args[4:])
            f_t=f_t11+f_t12+f_t21+f_t22
            if f_t==0: return np.nan #stopError
            p1=(f_t11+f_t12)/f_t
            p2=(f_t21+f_t22)/f_t
            llh-=log(f_t)
        pd.DataFrame(args).to_excel('./output/init_mle/arg.xlsx')
        return llh



    def constraint1_eq(self, args):
        return args[0]+args[1]-1.
    def constraint2_eq(self,args):
        return args[2]+args[3]-1.
    def optimz(self, args):
        from scipy.optimize import minimize

        b1=(0,  1)
        b2=(-100, 100) #from -100% to 100%
        b3=(0.000001, 100)
        bnds=tuple([b1]*4 +[b2]*(self.k*4)+[b3]*(self.k*2))
        cons=({'type':'eq', 'fun':self.constraint1_eq},
              {'type':'eq', 'fun':self.constraint2_eq})
        sol=minimize(self.obj, args, method='SLSQP', bounds=bnds, constraints=cons)
        return sol





if __name__=="__main__":
    args=[0.5,0.5,0.5,0.5]
    tns=63
    ins=MLE(tns)

    df=ins.r.iloc[:tns]
    kappa=[0]*(2*ins.k)
    gamma=[0.]*(ins.k*2)
    sigma= list(df.std().values**2) +list(df.std().values**2+0.01)
    args=args+kappa+gamma+sigma


    print(ins.obj(args))
    print(ins.optimz(args))


    pd.DataFrame(df.mean().values).to_excel('./output/init_mle/1_mean.xlsx')
    pd.DataFrame(df.std().values**2).to_excel('./output/init_mle/1_var.xlsx')











