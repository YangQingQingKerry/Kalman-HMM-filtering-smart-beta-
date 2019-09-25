#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import Config
import pandas as pd
import numpy as np
import os
class FilEst(Config):
    def __init__(self,state, order, tns, dval, diff,freq):
        #Dataset
        self.dir_path="./data/pca/keyFactor"
        self.df=pd.read_excel(self.dir_path+os.sep+'keyFactor_2009-09-11.xlsx')
        self.df.set_index('Date', inplace=True)
        self.rtn=100*(self.df/self.df.shift(1)-1)
        self.pred_rtn=pd.DataFrame(columns=self.df.columns)
        self.pred_df=pd.DataFrame(columns=self.df.columns)
        self.diff=diff
        self.f=100*(self.df.shift(-self.diff)/self.df-1).dropna()
        super(FilEst, self).__init__(self.f, state, order, tns)
        #Model Parameters
        self.D=[dval]*self.k
        self.freq=freq
        #Algo results
        self.pf=pd.DataFrame(columns=self.f.columns)
        self.p_path=[]
        self.modp=[]

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
                f_tp=(prob.T*theta).A[0,:]*self.kf.loc[t-1]+(prob.T*mu).A[0,:]
                sig_tp=(prob.T*theta).A[0,:]**2*self.Sigma+(prob.T*mu2).A[0,:]-(prob.T*mu).A[0,:]**2+ (prob.T*sigma2).A[0,:]
                psi=sig_tp/(sig_tp+self.D)
                self.kf.loc[t]=f_tp+psi*(self.f.iloc[t+self.tns-1].values-f_tp)
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
            op=(E*self.p).A[:,0]/np.sum(E*self.p)
            m=E/np.sum(E*self.p)
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
                        self.args[j*self.state**2+i]=J[_r,_s,_t]/O2[_s,_t]
        else:
            for _r in range(self.state):
                for _s in range(self.state):
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
            k.append(self.args[ix+i*self.k:ix+(i+1)*self.k])
            m.append(self.args[ix+(i+self.state)*self.k:ix+(i+1+self.state)*self.k])
            a=T_sq[i]+k[i]**2*T1_sq[i]+m[i]**2*O[0,i]-2*k[i]*TT1[i]-2*m[i]*T[i]+2*m[i]*k[i]*T1[i]
            self.args[ix+i*self.k:ix+(i+1)*self.k]=(TT1[i]-m[i]*T1[i])/T1_sq[i]
            self.args[ix+(i+self.state)*self.k:ix+(i+1+self.state)*self.k]=(T[i]-k[i]*T1[i])/O[0,i]
            self.args[ix+(2*self.state+i)*self.k:ix+(2*self.state+i+1)*self.k]= np.array(list(map(lambda x: max(x, 0.0001),a/O[0,i])))
        print("O:------")
        print(O)
        if self.order==2:
            print("O2:-----")
            print(O2)
        print("============UPDATE COMPLETED!==============")

    def Algo(self, period, window, Kalman=False):
        self.kalman(0, False, Kalman)
        self.hmm(0, Kalman)
        for t in range(1, period):
            if t%self.freq==0:
                self.kalman(t, False, Kalman)
                self.hmm(t, Kalman)
                print("The {}_th Update Begin!".format(t//self.freq))
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

            self.p_path.append(self.p.A[:,0])
            self.modp.append(list(self.args))
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

        tf=self.f.index[self.tns+1:self.tns+period]
        td=self.df.index[self.tns+1+self.diff: self.tns+1+self.diff+period]
        self.pf.index=tf
        self.pred_df.index=td
        self.pred_rtn.index=td
        #retrive daily-return prediction from the period-return time series
        self.pred_df.to_excel('./output/KalmanEM/pred_df.xlsx'.format(self.order, self.state))
        self.pred_rtn.to_excel('./output/KalmanEM/pred_rtn.xlsx'.format(self.order, self.state))
        self.r_df=self.df[(self.df.index>=td[0]) &(self.df.index<=td[-1])]
        self.r_df.to_excel('./output/KalmanEM/r_df.xlsx'.format(self.order, self.state))
        self.r_rtn=self.rtn[(self.rtn.index>=td[0]) &(self.rtn.index<=td[-1])]
        self.r_rtn.to_excel('./output/KalmanEM/r_rtn.xlsx'.format(self.order, self.state))
        #period-return prediction and the corresponding model parameters
        self.pf.to_excel('./output/KalmanEM/pf.xlsx'.format(self.order, self.state))
        self.r=self.f[(self.f.index>=tf[0]) &(self.f.index<=tf[-1])]
        self.r.to_excel('./output/KalmanEM/r.xlsx'.format(self.order, self.state))
        pd.DataFrame(self.p_path, index=tf).to_excel('./output/KalmanEM/p.xlsx'.format(self.order,self.state))
        pd.DataFrame(self.modp, index=tf).to_excel('./output/KalmanEM/modp.xlsx'.format(self.order,self.state))

if __name__=="__main__":
    order=1 #you can also test higher order HMM here
    state=2
    tns=63
    diff=10
    freq=10
    dval=0.0001
    ins=FilEst(state, order, tns,dval, diff,freq)
    ins.conFig()
    ins.Algo(period=len(ins.f)-tns, window=252, Kalman=True)










