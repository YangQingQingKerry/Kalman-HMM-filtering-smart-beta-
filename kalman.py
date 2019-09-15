#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from  initial_MLE import MLE
import pandas as pd
import numpy as np
class FilEst(MLE):
    def __init__(self,tns, val):
        super(FilEst, self).__init__(tns)
        self.pr=pd.DataFrame(columns=self.r.columns)
        self.p_path=[]
        self.D_init=np.array([val]*self.k)
        self.args= pd.read_excel('./output/init_mle/arg.xlsx')
        self.args=self.args[0].values

    def phik(self, s, v1, v0):
        from numpy.linalg import inv, det
        from math import exp, sqrt

        Sig=np.diag(self.args[4+(4+s)*self.k: 4+(5+s)*self.k])
        D=inv(Sig)
        x=v1-self.args[4+s*self.k:4+(1+s)*self.k]*v0-self.args[4+(2+s)*self.k:4+(3+s)*self.k]
        phi=exp(-0.5*np.sum(x*D*x.reshape(self.k,1))+0.5*np.dot(v1,v1))/sqrt(det(Sig))
        return phi

    def kalman(self, t):
        if t==0:
            self.kr=pd.DataFrame(columns=self.r.columns)
            self.kr.loc[0]=self.r.iloc[self.tns-1].values
            self.Sigma=self.D_init
            self.D=self.D_init
        else:
            kappa=self.args[4:4+2*self.k].reshape(2, self.k)
            gamma=self.args[4+2*self.k:4+4*self.k].reshape(2, self.k)
            gamma_2=(self.args[4+2*self.k:4+4*self.k]**2).reshape(2, self.k)
            eta_2=(self.args[4+4*self.k:]**2).reshape(2, self.k)
            r_tp=(self.p.T*kappa).A[0,:]*self.kr.loc[t-1]+(self.p.T*gamma).A[0,:]
            sig_tp=(self.p.T*kappa).A[0,:]**2*self.Sigma+(self.p.T*gamma_2).A[0,:]-(self.p.T*gamma).A[0,:]**2+ (self.p.T*eta_2).A[0,:]
            psi=sig_tp/(sig_tp+self.D)
            self.kr.loc[t]=r_tp+psi*(self.r.iloc[t+self.tns-1].values-r_tp)
            self.Sigma=self.D*psi
            #self.D=((self.r.iloc[self.tns-1:self.tns+t].values-self.kr.iloc[:t+1].values)**2).sum(axis=0)/t


    def hmm(self, t, Kalman=False):
        if t==0:
            #initialization
            self.P=np.matrix([self.args[:4]]).reshape(2,2).T
            self.p=np.matrix([self.args[2]/(1-self.args[0]+self.args[2]),self.args[1]/(1-self.args[3]+self.args[1])]).reshape(2,1)
            self.rho_J=np.array([0.]*8).reshape(2,2,2)
            self.rho_O=np.matrix([0.]*4).reshape(2,2)
            self.rho_T1=np.array([0.]*4*self.k).reshape(2,2,self.k)
            self.rho_T2=np.array([0.]*4*self.k).reshape(2,2,self.k)
            self.rho_t1=np.array([0.]*4*self.k).reshape(2,2,self.k)
            self.rho_t2=np.array([0.]*4*self.k).reshape(2,2,self.k)
            self.rho_Tt=np.array([0.]*4*self.k).reshape(2,2,self.k)
        else:
            #update
            if Kalman:
                v1=self.kr.iloc[t].values
                v0=self.kr.iloc[t-1].values
            else:
                v1=self.r.iloc[t+self.tns-1].values
                v0=self.r.iloc[t+self.tns-2].values
            E=np.asmatrix(np.diag([self.phik(0,v1,v0), self.phik(1,v1,v0)]))
            op=(E*self.p).A[:,0]/np.sum(E*self.p)
            if op[0]==0:
                m=np.asmatrix(np.diag([0., 1/self.p[1,0]]))
            elif op[1]==0:
                m=np.asmatrix(np.diag([1/self.p[0,0], 0.]))
            else:
                m=np.asmatrix(np.diag([op[0]+E[0,0]*op[1]/E[1,1], op[1]+E[1,1]*op[0]/E[0,0]]))
            self.rho_J[:,0,0]=np.dot(self.P*m, self.rho_J[:,0,0])+op[0]*self.P[0,0]*np.eye(2)[:,0]
            self.rho_J[:,0,1]=np.dot(self.P*m, self.rho_J[:,0,1])+op[0]*self.P[1,0]*np.eye(2)[:,1]
            self.rho_J[:,1,0]=np.dot(self.P*m, self.rho_J[:,1,0])+op[1]*self.P[0,1]*np.eye(2)[:,0]
            self.rho_J[:,1,1]=np.dot(self.P*m, self.rho_J[:,1,1])+op[1]*self.P[1,1]*np.eye(2)[:,1]
            self.rho_O[:,0]=self.P*m*self.rho_O[:,0]+op[0]*self.P[:,0]
            self.rho_O[:,1]=self.P*m*self.rho_O[:,1]+op[1]*self.P[:,1]
            self.rho_T1[:,0,:]=np.dot(self.P*m, self.rho_T1[:,0])+op[0]*(self.P[:,0]*v1.reshape(1,self.k)).A
            self.rho_T1[:,1,:]=np.dot(self.P*m, self.rho_T1[:,1])+op[1]*(self.P[:,1]*v1.reshape(1,self.k)).A
            self.rho_T2[:,0,:]=np.dot(self.P*m, self.rho_T2[:,0,:])+op[0]*(self.P[:,0]*(v1**2).reshape(1,self.k)).A
            self.rho_T2[:,1,:]=np.dot(self.P*m, self.rho_T2[:,1,:])+op[1]*(self.P[:,1]*(v1**2).reshape(1,self.k)).A
            self.rho_t1[:,0,:]=np.dot(self.P*m, self.rho_t1[:,0,:])+op[0]*(self.P[:,0]*v0.reshape(1,self.k)).A
            self.rho_t1[:,1,:]=np.dot(self.P*m, self.rho_t1[:,1,:])+op[1]*(self.P[:,1]*v0.reshape(1,self.k)).A
            self.rho_t2[:,0,:]=np.dot(self.P*m, self.rho_t2[:,0,:])+op[0]*(self.P[:,0]*(v0**2).reshape(1,self.k)).A
            self.rho_t2[:,1,:]=np.dot(self.P*m, self.rho_t2[:,1,:])+op[1]*(self.P[:,1]*(v0**2).reshape(1,self.k)).A
            self.rho_Tt[:,0,:]=np.dot(self.P*m, self.rho_Tt[:,0,:])+op[0]*(self.P[:,0]*(v0*v1).reshape(1,self.k)).A
            self.rho_Tt[:,1,:]=np.dot(self.P*m, self.rho_Tt[:,1,:])+op[1]*(self.P[:,1]*(v0*v1).reshape(1,self.k)).A
            self.p=self.P*op.reshape(2,1)


    def EM(self, lam):
        args_old=self.args.copy()
        k1=args_old[4:4+self.k]
        k2=args_old[4+self.k:4+2*self.k]
        g1=args_old[4+2*self.k:4+3*self.k]
        g2=args_old[4+3*self.k:4+4*self.k]
        e1=args_old[4+4*self.k:4+5*self.k]
        e2=args_old[4+5*self.k:4+6*self.k]
        #transition prob
        J=self.rho_J.sum(axis=0)
        O=self.rho_O.sum(axis=0)
        self.args[0]=lam*self.args[0]+(1-lam)*J[0,0]/O[0,0]
        self.args[1]=lam*self.args[1]+(1-lam)*J[0,1]/O[0,0]
        self.args[2]=lam*self.args[2]+(1-lam)*J[1,0]/O[0,1]
        self.args[3]=lam*self.args[3]+(1-lam)*J[1,1]/O[0,1]
        #kappa
        Tt=self.rho_Tt.sum(axis=0)
        t1=self.rho_t1.sum(axis=0)
        t2=self.rho_t2.sum(axis=0)
        self.args[4:4+self.k]=lam*k1+(1-lam)*(Tt[0]-g1*t1[0])/t2[0]
        self.args[4+self.k:4+2*self.k]=lam*k2+(1-lam)*(Tt[1]-g2*t1[1])/t2[1]
        #gamma
        T1=self.rho_T1.sum(axis=0)
        k1_new=self.args[4:4+self.k]
        k2_new=self.args[4+self.k:4+2*self.k]
        self.args[4+2*self.k:4+3*self.k]=lam*g1+(1-lam)*(T1[0]-k1_new*t1[0])/O[0,0]
        self.args[4+3*self.k:4+4*self.k]=lam*g2+(1-lam)*(T1[1]-k2_new*t1[1])/O[0,1]
        #eta
        T2=self.rho_T2.sum(axis=0)
        g1_new=self.args[4+2*self.k:4+3*self.k]
        g2_new=self.args[4+3*self.k:4+4*self.k]
        a1=T2[0]-2*g1_new*T1[0]+g1_new**2*O[0,0]+k1_new**2*t2[0]-2*k1_new*Tt[0]+2*k1_new*g1_new*t1[0]
        a2=T2[1]-2*g2_new*T1[1]+g2_new**2*O[0,1]+k2_new**2*t2[1]-2*k2_new*Tt[1]+2*k2_new*g2_new*t1[1]
        self.args[4+4*self.k:4+5*self.k]=lam*e1+(1-lam)*a1/O[0,0]
        self.args[4+5*self.k:4+6*self.k]=lam*e2+(1-lam)*a2/O[0,1]







    def Estimate(self, freq, period, lam, Kalman):
        pd.DataFrame(self.args).to_excel('arg/args_{}.xlsx'.format(1))
        if Kalman:
            for t in range(1, period):
                if (t%freq==0) and (t>1):
                    self.EM(lam)
                    print("{}_th Update!".format(t//freq))
                    pd.DataFrame(self.args).to_excel('arg/args_{}.xlsx'.format(t))
                    self.kalman(0)
                    self.hmm(0, True)
                    _=0  #counter
                    while _<t:
                        _+=1
                        self.kalman(_)
                        self.hmm(_,True)
                else:
                    self.kalman(t)
                    self.hmm(t, True)
                self.p_path.append(self.p[0,0])
                kappa=self.args[4:4+2*self.k].reshape(2, self.k)
                gamma=self.args[4+2*self.k:4+4*self.k].reshape(2, self.k)
                self.pr.loc[t-1]=(self.p.T*kappa).A[0,:]*self.kr.loc[t]+(self.p.T*gamma).A[0,:]
        else:
            for t in range(1, period):
                if (t%freq==0) and (t>1):
                    self.EM(lam)
                    print("{}_th Update!".format(t//freq))
                    pd.DataFrame(self.args).to_excel('arg/args_{}.xlsx'.format(t))
                    self.hmm(0)
                    _=0
                    while _<t:
                        _+=1
                        self.hmm(_)
                else:
                    self.hmm(t)
                self.p_path.append(self.p[0,0])
                kappa=self.args[4:4+2*self.k].reshape(2, self.k)
                gamma=self.args[4+2*self.k:4+4*self.k].reshape(2, self.k)
                self.pr.loc[t-1]=(self.p.T*kappa).A[0,:]*self.r.iloc[t+self.tns-1]+(self.p.T*gamma).A[0,:]

        index=self.r.index.tolist()[self.tns+1: self.tns+period]
        self.pr.index=index
        self.pr.to_excel('pr.xlsx')
        self.r_s=self.r[(self.r.index>=index[0]) & (self.r.index<=index[-1])]
        self.r_s.to_excel('r.xlsx')
        pd.DataFrame(self.p_path).to_excel('p.xlsx')

if __name__=="__main__":
    ins2=FilEst(tns=63, val=0.00003)
    ins2.hmm(0)
    ins2.kalman(0)
    ins2.Estimate(freq=3, period=189, lam=0.94, Kalman=True)







