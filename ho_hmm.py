#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import Config
import pandas as pd
import numpy as np

class FilEst(Config):
    def __init__(self, url_factor, state, order, tns, dval, diff,freq):
        self.df=pd.read_excel(url_factor)
        self.df.set_index('Datetime', inplace=True)
        self.rtn=100*(self.df/self.df.shift(1)-1)
        self.pred_rtn=pd.DataFrame(columns=self.df.columns)
        self.pred_df=pd.DataFrame(columns=self.df.columns)
        self.diff=diff
        self.f=100*(self.df/self.df.shift(self.diff)-1).dropna()
        super(FilEst, self).__init__(self.f, state, order, tns)
        self.D=[dval]*self.k
        self.freq=freq
        self.pf=pd.DataFrame(columns=self.f.columns)


    def phik(self, s, v, v_1):
        from numpy.linalg import inv, det
        from math import exp, sqrt

        ix=self.state**(2*self.order)
        Sig=np.diag(self.args[ix+(2*self.state+s)*self.k: ix+(2*self.state+s+1)*self.k])
        D=inv(Sig)
        x=v-self.args[ix+s*self.k:ix+(1+s)*self.k]*v_1-self.args[ix+(self.state+s)*self.k:ix+(self.state+1+s)*self.k]
        phi=exp(-0.5*np.sum(x*D*x.reshape(self.k,1))+0.5*np.dot(v,v))/sqrt(det(Sig))
        return phi

    def kalman(self, t, restart=False, Kalman=False):
        if Kalman:
            if (t==0) | restart:
                if t==0:
                    self.kf=pd.DataFrame(columns=self.f.columns)
                    self.kf.loc[0]=self.f.iloc[self.tns-1].values
                    self.Sigma=self.D
                    self.loc=0
                else:
                    self.kf.loc[t]=self.f.iloc[t+self.tns-1]
                    self.loc=t
            else:
                ix=self.state**(2*self.order)
                theta=self.args[ix:ix+self.state*self.k].reshape(self.state, self.k)
                mu=self.args[ix+self.state*self.k:ix+2*self.state*self.k].reshape(self.state, self.k)
                mu2=(self.args[ix+self.state*self.k:ix+2*self.state*self.k]**2).reshape(self.state, self.k)
                sigma2=self.args[ix+2*self.state*self.k:].reshape(self.state, self.k)
                if self.order==2:
                    prob=self.p.reshape(self.state, self.state).sum(axis=1)
                else:
                    prob=self.p
                self.f_tp=(prob.T*theta).A[0,:]*self.kf.loc[t-1]+(prob.T*mu).A[0,:]
                self.sig_tp=(prob.T*theta).A[0,:]**2*self.Sigma+(prob.T*mu2).A[0,:]-(prob.T*mu).A[0,:]**2+ (prob.T*sigma2).A[0,:]
                psi=self.sig_tp/(self.sig_tp+self.D)
                self.kf.loc[t]=self.f_tp+psi*(self.f.iloc[t+self.tns-1].values-self.f_tp)
                self.Sigma=self.D*psi
                self.D=((self.f.iloc[self.tns+self.loc-1:t+self.tns].values-self.kf.iloc[self.loc:t+1].values)**2).sum(axis=0)/t
        else:
            pass


    def hmm(self, t, Kalman):
        if t==0:
            self.P=np.matrix([self.args[:self.state**(2*self.order)]]).reshape(self.state**self.order,self.state**self.order)
            self.p=np.matrix([1/self.state**self.order]*self.state**self.order).reshape(self.state**self.order,1)
            if self.order==2:
                self.rho_J=np.array([0.]*self.state**5).reshape(self.state**2,self.state,self.state, self.state)
                self.rho_O2=np.array([0.]*self.state**4).reshape(self.state**2,self.state,self.state)
            else: #order==1
                self.rho_J=np.array([0.]*self.state**3).reshape(self.state,self.state,self.state)
            self.rho_O=np.matrix([0.]*self.state**(self.order+1)).reshape(self.state**self.order,self.state)
            self.rho_T=np.array([0.]*self.state**(self.order+1)*self.k).reshape(self.state**self.order,self.state,self.k)
            self.rho_T_sq=np.array([0.]*self.state**(self.order+1)*self.k).reshape(self.state**self.order,self.state,self.k)
            self.rho_T1=np.array([0.]*self.state**(self.order+1)*self.k).reshape(self.state**self.order,self.state,self.k)
            self.rho_T1_sq=np.array([0.]*self.state**(self.order+1)*self.k).reshape(self.state**self.order,self.state,self.k)
            self.rho_TT1=np.array([0.]*self.state**(self.order+1)*self.k).reshape(self.state**self.order,self.state,self.k)
        else:
            if Kalman:
                v=self.kf.iloc[t].values
                v_1=self.kf.iloc[t-1].values
            else:
                v=self.f.iloc[t+self.tns-1].values
                v_1=self.f.iloc[t+self.tns-2].values
            if self.order==2:
                E=np.asmatrix(np.diag([self.phik(i,v,v_1) for i in range(self.state) for j in range(self.state)]))
            else:
                E=np.asmatrix(np.diag([self.phik(i,v,v_1) for i in range(self.state)]))
            #in case of E=diag([0,0,...,0])
            if np.sum(E*self.p)>0:
                op=(E*self.p).A[:,0]/np.sum(E*self.p)
                m=E/np.sum(E*self.p)
            else:
                op=np.array([0.]*self.state**self.order)
                m=np.asmatrix(np.diag([1.]*self.state**self.order))

            for _r in range(self.state):
                if self.order==2:
                    for _s in range(self.state):
                        for _t in range(self.state):
                            #i->j
                            i=_s*self.state+_t
                            j=_r*self.state+_s
                            self.rho_J[:,_r,_s,_t]=np.dot(self.P*m, self.rho_J[:,_r,_s,_t])+op[i]*self.P[j,i]*np.eye(self.state**2)[:,j]
                        #rho_O2
                        i=_r*self.state+_s
                        self.rho_O2[:,_r,_s]=np.dot(self.P*m, self.rho_O2[:,_r,_s])+op[i]*self.P.A[:,i]
                else:
                    for _s in range(self.state):
                        self.rho_J[:,_r,_s]=np.dot(self.P*m, self.rho_J[:,_r,_s])+op[_s]*self.P[_r,_s]*np.eye(self.state)[:,_r]
                #rho_O, rho_T
                if self.order==2:
                    self.rho_O[:,_r]=self.P*m*self.rho_O[:,_r]+sum([op[_r*self.state+_s]*self.P[:,_r*self.state+_s] for _s in range(self.state)])
                    self.rho_T[:,_r,:]=np.dot(self.P*m, self.rho_T[:,_r,:])+(sum([op[_r*self.state+_s]*self.P[:,_r*self.state+_s] for _s in range(self.state)])*v.reshape(1,self.k)).A
                    self.rho_T1[:,_r,:]=np.dot(self.P*m, self.rho_T1[:,_r,:])+(sum([op[_r*self.state+_s]*self.P[:,_r*self.state+_s] for _s in range(self.state)])*v_1.reshape(1,self.k)).A
                    self.rho_T_sq[:,_r,:]=np.dot(self.P*m, self.rho_T_sq[:,_r,:])+(sum([op[_r*self.state+_s]*self.P[:,_r*self.state+_s] for _s in range(self.state)])*(v**2).reshape(1,self.k)).A
                    self.rho_T1_sq[:,_r,:]=np.dot(self.P*m, self.rho_T1_sq[:,_r,:])+(sum([op[_r*self.state+_s]*self.P[:,_r*self.state+_s] for _s in range(self.state)])*(v_1**2).reshape(1,self.k)).A
                    self.rho_TT1[:,_r,:]=np.dot(self.P*m, self.rho_TT1[:,_r,:])+(sum([op[_r*self.state+_s]*self.P[:,_r*self.state+_s] for _s in range(self.state)])*(v*v_1).reshape(1,self.k)).A
                else: #order==1
                    self.rho_O[:,_r]=self.P*m*self.rho_O[:,_r]+ op[_r]*self.P[:,_r]
                    self.rho_T[:,_r,:]=np.dot(self.P*m, self.rho_T[:,_r,:])+op[_r]*(self.P[:,_r]*v.reshape(1,self.k)).A
                    self.rho_T1[:,_r,:]=np.dot(self.P*m, self.rho_T1[:,_r,:])+op[_r]*(self.P[:,_r]*v_1.reshape(1,self.k)).A
                    self.rho_T_sq[:,_r,:]=np.dot(self.P*m, self.rho_T_sq[:,_r,:])+op[_r]*(self.P[:,_r]*(v**2).reshape(1,self.k)).A
                    self.rho_T1_sq[:,_r,:]=np.dot(self.P*m, self.rho_T1_sq[:,_r,:])+op[_r]*(self.P[:,_r]*(v_1**2).reshape(1,self.k)).A
                    self.rho_TT1[:,_r,:]=np.dot(self.P*m, self.rho_TT1[:,_r,:])+op[_r]*(self.P[:,_r]*(v*v_1).reshape(1,self.k)).A
            self.p=self.P*op.reshape(self.state**self.order,1)

    def EM(self):
        J=self.rho_J.sum(axis=0)
        O=self.rho_O.sum(axis=0)
        if self.order==2:
            O2=self.rho_O2.sum(axis=0)
            for _r in range(self.state):
                for _s in range(self.state):
                    for _t in range(self.state):
                        #i->j
                        i=_s*self.state+_t
                        j=_r*self.state+_s
                        if O2[_s, _t]==0:
                            self.args[j*self.state**2+i]= 0
                        else:
                            self.args[j*self.state**2+i]=J[_r,_s,_t]/O2[_s,_t]
        else:
            for _r in range(self.state):
                for _s in range(self.state):
                    if O[0,_s]==0:
                        self.args[_r*self.state+_s]=0
                    else:
                        self.args[_r*self.state+_s]=J[_r,_s]/O[0,_s]

        T=self.rho_T.sum(axis=0)
        T1=self.rho_T1.sum(axis=0)
        T_sq=self.rho_T_sq.sum(axis=0)
        T1_sq=self.rho_T1_sq.sum(axis=0)
        TT1=self.rho_TT1.sum(axis=0)
        ix=self.state**(2*self.order)
        #args_old=self.args.copy()
        k=[]; m=[];
        for i in range(self.state):
            if O[0,i]==0:
                k.append(self.args[ix+i*self.k:ix+(i+1)*self.k])
                m.append(self.args[ix+(i+self.state)*self.k:ix+(i+1+self.state)*self.k])
            else:
                k.append(self.args[ix+i*self.k:ix+(i+1)*self.k])
                m.append(self.args[ix+(i+self.state)*self.k:ix+(i+1+self.state)*self.k])
                a=T_sq[i]+k[i]**2*T1_sq[i]+m[i]**2*O[0,i]-2*k[i]*TT1[i]-2*m[i]*T[i]+2*m[i]*k[i]*T1[i]
                self.args[ix+i*self.k:ix+(i+1)*self.k]=(TT1[i]-m[i]*T1[i])/T1_sq[i]
                self.args[ix+(i+self.state)*self.k:ix+(i+1+self.state)*self.k]=(T[i]-k[i]*T1[i])/O[0,i]
                self.args[ix+(2*self.state+i)*self.k:ix+(2*self.state+i+1)*self.k]= np.array(list(map(lambda x: max(x, 0.0001),a/O[0,i])))


    def Algo(self, period, window, Kalman=False):
        self.kalman(0, False, Kalman)
        self.hmm(0, Kalman)
        for t in range(1, period):
            if t%self.freq==0:
                self.kalman(t, False, Kalman)
                self.hmm(t, Kalman)
                self.EM()
                self.hmm(0, Kalman)  #restart HMM filter
                _=max(1, t-window+1)
                self.kalman(_-1, True, Kalman)
                while _<=t:
                    self.kalman(_, False, Kalman)
                    self.hmm(_, Kalman)
                    _+=1

            else:
                self.kalman(t,False, Kalman)
                self.hmm(t, Kalman)
            if self.order==2:
                prob=self.p.reshape(self.state, self.state).sum(axis=1)
            else:
                prob=self.p
            ix=self.state**(2*self.order)
            theta=self.args[ix:ix+self.state*self.k].reshape(self.state,self.k)
            mu=self.args[ix+self.state*self.k:ix+2*self.state*self.k].reshape(self.state, self.k)
            if Kalman:
                self.pf.loc[t-1]=(prob.T*theta).A[0]*self.kf.iloc[-1].values+(prob.T*mu).A[0]
            else:
                self.pf.loc[t-1]=(prob.T*theta).A[0]*self.f.iloc[t+self.tns-1].values+(prob.T*mu).A[0]
            self.pred_df.loc[t-1]=(1+0.01*self.pf.loc[t-1])*self.df.iloc[t+self.tns]
            self.pred_rtn.loc[t-1]=100*(self.pred_df.loc[t-1]/self.df.iloc[t+self.tns+self.diff-1]-1)

        '''
        Last-step prediction
        '''
        t=period
        Sig=self.args[ix+2*self.state*self.k:ix+3*self.state*self.k].reshape(self.state, self.k)
        if Kalman:
            self.kalman(t, False, Kalman)
            pf_pred=(prob.T*theta).A[0]*self.kf.iloc[-1].values+(prob.T*mu).A[0]
        else:
            pf_pred=(prob.T*theta).A[0]*self.f.iloc[t+self.tns-1].values+(prob.T*mu).A[0]
        df_pred=(1+0.01*pf_pred)*self.df.iloc[t+self.tns]
        rtn_pred=100*(df_pred/self.df.iloc[t+self.tns+self.diff-1]-1)
        if Kalman:
            self.fvar=self.sig_tp
            return rtn_pred, rtn_pred/np.sqrt(self.fvar)
        else:
            self.fvar=(prob.T*Sig).A[0]
            return rtn_pred, rtn_pred/np.sqrt(self.fvar)









