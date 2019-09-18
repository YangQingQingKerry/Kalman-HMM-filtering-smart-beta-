#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from  initial_MLE import MLE
import pandas as pd
import numpy as np
class FilEst(MLE):
    def __init__(self,state, tns, val, rollowing_window):
        super(FilEst, self).__init__(state, tns)
        self.rollowing_window=rollowing_window
        self.pf=pd.DataFrame(columns=self.f.columns)
        self.p_path=[]
        self.D=np.array([val]*self.k)
        self.args= pd.read_excel('./output/init_mle/arg_{}state.xlsx'.format(self.state))
        self.args=self.args[0].values
        self.mp=[]
        self.names=["p_{}{}".format(i,j) for i in range(self.state) for j in range(self.state)]+\
                   ['kappa_s{}_f{}'.format(i,j+1) for i in range(self.state) for j in range(self.k)]+\
                   ['gamma_s{}_f{}'.format(i,j+1) for i in range(self.state) for j in range(self.k)]+\
                   ['eta2_s{}_f{}'.format(i,j+1) for i in range(self.state) for j in range(self.k)]

    def phik(self, s, v1, v0):
        from numpy.linalg import inv, det
        from math import exp, sqrt
        _=self.state**2
        Sig=np.diag(self.args[_+(2*self.state+s)*self.k: _+(2*self.state+s+1)*self.k])
        D=inv(Sig)
        x=v1-self.args[_+s*self.k:_+(1+s)*self.k]*v0-self.args[_+(self.state+s)*self.k:_+(self.state+s+1)*self.k]
        phi=exp(-0.5*np.sum(x*D*x.reshape(self.k,1))+0.5*np.dot(v1,v1))/sqrt(det(Sig))
        return phi

    def kalman(self, t, start_loc):
        if t==0:
            self.kf=pd.DataFrame(columns=self.f.columns)
            self.kf.loc[0]=self.f.iloc[start_loc-1].values
            self.Sigma=self.D
        else:
            _=self.state**2
            kappa=self.args[_:_+self.state*self.k].reshape(self.state, self.k)
            gamma=self.args[_+self.state*self.k:_+2*self.state*self.k].reshape(self.state, self.k)
            gamma_2=(self.args[_+self.state*self.k:_+2*self.state*self.k]**2).reshape(self.state, self.k)
            eta_2=(self.args[_+2*self.state*self.k:]**2).reshape(self.state, self.k)
            f_tp=(self.p.T*kappa).A[0,:]*self.kf.loc[t-1]+(self.p.T*gamma).A[0,:]
            sig_tp=(self.p.T*kappa).A[0,:]**2*self.Sigma+(self.p.T*gamma_2).A[0,:]-(self.p.T*gamma).A[0,:]**2+ (self.p.T*eta_2).A[0,:]
            psi=sig_tp/(sig_tp+self.D)
            self.kf.loc[t]=f_tp+psi*(self.f.iloc[t+start_loc-1].values-f_tp)
            self.Sigma=self.D*psi
            self.D=((self.f.iloc[start_loc-1:start_loc+t].values-self.kf.iloc[:t+1].values)**2).sum(axis=0)/t

    def hmm(self, t, start_loc, Kalman=False):
        if t==0:
            #initialization
            self.P=np.matrix([self.args[:self.state**2]]).reshape(self.state,self.state).T
            self.p=np.matrix([1/self.state]*self.state).reshape(self.state,1)
            self.rho_J=np.array([0.]*self.state**3).reshape(self.state,self.state,self.state)
            self.rho_O=np.matrix([0.]*self.state**2).reshape(self.state,self.state)
            self.rho_T1=np.array([0.]*self.state**2*self.k).reshape(self.state,self.state,self.k)
            self.rho_T2=np.array([0.]*self.state**2*self.k).reshape(self.state,self.state,self.k)
            self.rho_t1=np.array([0.]*self.state**2*self.k).reshape(self.state,self.state,self.k)
            self.rho_t2=np.array([0.]*self.state**2*self.k).reshape(self.state,self.state,self.k)
            self.rho_Tt=np.array([0.]*self.state**2*self.k).reshape(self.state,self.state,self.k)
        else:
            #update
            if Kalman:
                v1=self.kf.iloc[t].values
                v0=self.kf.iloc[t-1].values
            else:
                v1=self.f.iloc[t+start_loc-1].values
                v0=self.f.iloc[t+start_loc-2].values
            if self.state==1:
                op=np.matrix([[1.]])
                m=np.matrix([[1.0]])
            else:
                E=np.asmatrix(np.diag([self.phik(i,v1,v0) for i in range(self.state)]))
                op=(E*self.p).A[:,0]/np.sum(E*self.p)
                m=E/np.sum(E*self.p)
            for i in range(self.state):
                #rho_J
                for j in range(self.state):
                    self.rho_J[:,i,j]=np.dot(self.P*m, self.rho_J[:,i,j])+op[i]*self.P[j,i]*np.eye(self.state)[:,j]
                #rho_O, rho_T
                self.rho_O[:,i]=self.P*m*self.rho_O[:,i]+op[i]*self.P[:,i]
                self.rho_T1[:,i,:]=np.dot(self.P*m, self.rho_T1[:,i,:])+op[i]*(self.P[:,i]*v1.reshape(1,self.k)).A
                self.rho_T2[:,i,:]=np.dot(self.P*m, self.rho_T2[:,i,:])+op[i]*(self.P[:,i]*(v1**2).reshape(1,self.k)).A
                self.rho_t1[:,i,:]=np.dot(self.P*m, self.rho_t1[:,i,:])+op[i]*(self.P[:,i]*v0.reshape(1,self.k)).A
                self.rho_t2[:,i,:]=np.dot(self.P*m, self.rho_t2[:,i,:])+op[i]*(self.P[:,i]*(v0**2).reshape(1,self.k)).A
                self.rho_Tt[:,i,:]=np.dot(self.P*m, self.rho_Tt[:,i,:])+op[i]*(self.P[:,i]*(v0*v1).reshape(1,self.k)).A
            self.p=self.P*op.reshape(self.state,1)

    def EM(self, lam):
        args_old=self.args.copy()
        k_old=[]
        g_old=[]
        e_old=[]
        _=self.state**2
        for i in range(self.state):
            k_old.append(args_old[_+i*self.k:_+(i+1)*self.k])
            g_old.append(args_old[_+(self.state+i)*self.k:_+(self.state+i+1)*self.k])
            e_old.append(args_old[_+(2*self.state+i)*self.k:_+(2*self.state+i+1)*self.k])

        #transition prob
        J=self.rho_J.sum(axis=0)
        O=self.rho_O.sum(axis=0)
        for i in range(self.state):
            for j in range(self.state):
               self.args[self.state*i+j]=lam*self.args[self.state*i+j]+(1-lam)*J[i,j]/O[0,i]
        #other model parameters
        T1=self.rho_T1.sum(axis=0)
        T2=self.rho_T2.sum(axis=0)
        t1=self.rho_t1.sum(axis=0)
        t2=self.rho_t2.sum(axis=0)
        Tt=self.rho_Tt.sum(axis=0)
        for i in range(self.state):
            #kappa
            self.args[_+i*self.k: _+(i+1)*self.k]=lam*k_old[i]+(1-lam)*np.array(list(map(lambda x: max(x, 0.0001), (Tt[i]-g_old[i]*t1[i])/t2[i])))
            #gamma
            self.args[_+(self.state+i)*self.k:_+(self.state+i+1)*self.k]=lam*g_old[i]+(1-lam)*(T1[i]-k_old[i]*t1[i])/O[0,i]
            #eta
            a=T2[i]-2*g_old[i]*T1[i]+g_old[i]**2*O[0,i]+k_old[i]**2*t2[i]-2*k_old[i]*Tt[i]+2*k_old[i]*g_old[i]*t1[i]
            self.args[_+(2*self.state+i)*self.k: _+(2*self.state+i+1)*self.k]=lam*e_old[i]+(1-lam)*a/O[0,i]
        print("O:------")
        print(O)
        print("==============UPDATE COPLETED============")


    def Estimate(self, freq, period, lam, Kalman):
        '''drop the training dataset for initial_MLE'''
        start_loc=self.tns

        if Kalman:
            for t in range(1, period):
                if t%freq==0:
                    '''
                    Step 1:
                    ========
                        Based on Theta(t-1), kr[t] and the hmm filters at time t is computed
                    '''
                    if start_loc!=self.tns:
                        self.kalman(self.rollowing_window+freq, start_loc)
                        self.hmm(self.rollowing_window+freq, start_loc, Kalman=True)
                    else:
                        self.kalman(t, start_loc)
                        self.hmm(t, start_loc, Kalman=True)
                    '''
                    Step 2:
                    ========
                        Based on kr[0], , kr[t] and hmm filters, Theta(t) is calculated
                    '''
                    print("{}_th Update!".format(t//freq))
                    self.EM(lam)
                    '''
                    Step3:
                    ========
                        All kalman filters
                           kr[0], ..., kr[t]
                        and hmm filters
                           rho_J[0-t], ...
                        are updated according to Theta[t] over the new rollowing_window
                    '''
                    '''recalculate the start_loc of the rollowing window'''
                    start_loc=max(self.tns, t-self.rollowing_window+self.tns)
                    self.kalman(0, start_loc)
                    self.hmm(0, start_loc, Kalman=True)
                    _=0  #counter
                    while _<min(self.rollowing_window, t):
                        _+=1
                        self.kalman(_, start_loc)
                        self.hmm(_, start_loc, Kalman=True)
                else:
                    '''
                    Only update the kalman-hmm filters based on Theta(t-1)
                    '''
                    if start_loc!=self.tns:
                        self.kalman(self.rollowing_window+t%freq, start_loc)
                        self.hmm(self.rollowing_window+t%freq,  start_loc,  Kalman=True)
                    else:
                        self.kalman(t, start_loc)
                        self.hmm(t,  start_loc,  Kalman=True)
                self.mp.append(list(self.args))
                self.p_path.append(self.p.A[:,0])
                _=self.state**2
                kappa=self.args[_:_+self.state*self.k].reshape(self.state, self.k)
                gamma=self.args[_+self.state*self.k:_+2*self.state*self.k].reshape(self.state, self.k)
                self.pf.loc[t-1]=(self.p.T*kappa).A[0,:]*self.kf.iloc[-1]+(self.p.T*gamma).A[0,:]
            '''Here is the last step's prediction results'''
            self.mp.append(list(self.args))
            self.p_path.append(self.p.A[:,0])

        else:
            for t in range(1, period):
                if t%freq==0:
                    '''
                    Step 1:
                    ========
                        Based on Theta(t-1), kr[t] and the hmm filters at time t is computed
                    '''
                    if start_loc!=self.tns:
                        self.hmm(self.rollowing_window+freq, start_loc)
                    else:
                        self.hmm(t,start_loc)
                    '''
                    Step 2:
                    ========
                        Based on kr[0], , kr[t] and hmm filters, Theta(t) is calculated
                    '''
                    self.EM(lam)
                    print("{}_th Update!".format(t//freq))
                    '''
                    Step3:
                    ========
                        All kalman filters
                           kr[0], ..., kr[t]
                        and hmm filters
                           rho_J[0-t], ...
                        are updated according to Theta[t] over the new rollowing_window
                    '''
                    '''recalculate the start_loc of the rollowing window'''
                    start_loc=max(self.tns, t-self.rollowing_window+self.tns)
                    self.hmm(0, start_loc)
                    _=0
                    while _<min(self.rollowing_window, t):
                        _+=1
                        self.hmm(_, start_loc)
                else:
                    '''
                    Only update the kalman-hmm filters based on Theta(t-1)
                    '''
                    if start_loc!=self.tns:
                        self.hmm(self.rollowing_window+t%freq, start_loc)
                    else:
                        self.hmm(t, start_loc)
                self.mp.append(list(self.args))
                self.p_path.append(self.p.A[:,0])
                _=self.state**2
                kappa=self.args[_:_+self.state*self.k].reshape(self.state, self.k)
                gamma=self.args[_+self.state*self.k:_+2*self.state*self.k].reshape(self.state, self.k)
                self.pf.loc[t-1]=(self.p.T*kappa).A[0,:]*self.f.iloc[t+self.tns-1]+(self.p.T*gamma).A[0,:]
            '''Here is the last step's prediction results'''
            self.mp.append(list(self.args))
            self.p_path.append(self.p.A[:,0])

        index=self.f.index.tolist()[self.tns+1: self.tns+period]
        self.pf.index=index
        self.pf=self.pf*(self.max-self.min)+self.min
        self.pf.to_excel('./output/rt_{}state/pf.xlsx'.format(self.state))
        self.f_s=self.f[(self.f.index>=index[0]) & (self.f.index<=index[-1])]
        self.f_s=self.f_s*(self.max-self.min)+self.min
        self.f_s.to_excel('./output/rt_{}state/f.xlsx'.format(self.state))
        pd.DataFrame(self.p_path[1:]).to_excel('./output/rt_{}state/p.xlsx'.format(self.state))
        pd.DataFrame(self.mp[1:], index=index, columns=self.names).to_excel('./output/rt_{}state/mp.xlsx'.format(self.state))



if __name__=="__main__":
    state=1
    tns=63
    ins2=FilEst(state,tns, val=0.00001, rollowing_window=252)
    ins2.hmm(0, start_loc=tns)
    ins2.kalman(0, start_loc=tns)
    ins2.Estimate(freq=5, period=len(ins2.f)-tns, lam=0., Kalman=False)








