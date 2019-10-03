#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ho_hmm import FilEst
import pandas as pd
import numpy as np
import os


class BackTCell(FilEst):
    def __init__(self, dir_path, state, order, tns, dval, diff, freq, window,lamb, thresh):
        self.dir_path=dir_path
        self.state=state
        self.order=order
        self.tns=tns
        self.dval=dval
        self.diff=diff
        self.freq=freq
        self.window=window
        self.lamb=lamb
        self.thresh=thresh


    def PredictionCell(self, factor_url, Kalman):
        super(BackTCell, self).__init__(factor_url,self.state,self.order,self.tns,self.dval,self.diff,self.freq)
        self.conFig()
        return self.Algo(len(self.f)-tns, self.window, Kalman)

    def Obj(self, args):
        port_alpha=np.dot(args, 100*self.alpha.alpha.values)
        port_factor=np.dot(args, np.dot(self.BETA, self.prediction.values))
        h=np.dot(args, self.BETA)
        port_factor_var=np.dot(np.dot(h, np.diag(self.fvar)),h.T)
        h_epsilon=np.dot(args, self.epsilon)
        port_epsilon=np.dot(np.dot(h_epsilon, self.eigen),h_epsilon.T)
        rtn=port_alpha+port_factor
        var=self.lamb*(port_factor_var+port_epsilon)

        obj_fun=var-rtn
        return   obj_fun


    def cons(self, args):
        return np.sum(args)
    def cons_maker(self, i=0):
        def cons_ineq(args):
            return np.sign(self.scores[i])*np.dot(args,self.BETA[:,i])
        return cons_ineq

    def OptimizationCell(self, args):
        from scipy.optimize import minimize

        bnds=tuple([(-min(0.5, self.bmk[i]), 0.5) for i in range(len(self.bmk))])
        cons=[{'type':'eq', 'fun':self.cons}]
        for _, item in enumerate(self.scores):
            if abs(item)>=self.thresh:
                cons+=[{'type':'ineq', 'fun': self.cons_maker(i=_)}]
            else:
                cons+=[{'type':'eq', 'fun':self.cons_maker(i=_)}]
        cons=tuple(cons)
        sol=minimize(self.Obj, args, method='SLSQP', bounds=bnds, constraints=cons)
        return sol



    def Strgy(self, etf_url, Kalman):
        etf=pd.read_excel(etf_url)
        datelist=etf.Date.unique()

        for i, date in enumerate(datelist):
            t=str(date)
            factor_url=self.dir_path+os.sep+'keyFactor/keyFactor_{}-{}-{}.xlsx'.format(t[:4],t[4:6],t[6:])
            exp_url=self.dir_path+os.sep+'factorExposure/factorExposure_{}-{}-{}.xlsx'.format(t[:4], t[4:6],t[6:])
            alpha_url=self.dir_path+os.sep+'alpha/alpha_{}-{}-{}.xlsx'.format(t[:4], t[4:6],t[6:])
            epsilon_url=self.dir_path+os.sep+'epsilon/epsilon_{}-{}-{}.xlsx'.format(t[:4], t[4:6], t[6:])
            eigen_url=self.dir_path+os.sep+'eigen/eigen_{}-{}-{}.xlsx'.format(t[:4], t[4:6], t[6:])

            #load alpha
            self.alpha=pd.read_excel(alpha_url)
            self.alpha.columns=['Company','alpha']

            #load exposure
            self.beta=pd.read_excel(exp_url)
            self.beta.columns=['Company']+list(self.beta.columns[1:])
            self.BETA=self.beta[self.beta.columns[1:]].values

            #load epsilon and eigenvalues
            self.epsilon=pd.read_excel(epsilon_url)
            self.epsilon=self.epsilon[self.epsilon.columns[1:]].values
            self.eigen=pd.read_excel(eigen_url)
            self.eigen=self.eigen[self.eigen.columns[1:]].values

            #weight: take the whole universe into consideration
            Wgt=pd.DataFrame(index=self.beta.Company)
            etf_dy=etf[etf.Date==date][['Company','% Wgt']]
            etf_dy.set_index('Company', inplace=True)
            self.etf_dy=pd.merge(Wgt, etf_dy, left_index=True, right_index=True, how='outer').fillna(0).groupby('Company').sum()
            self.bmk=self.etf_dy['% Wgt'].values


            #call prediction function
            self.prediction, self.scores=self.PredictionCell(factor_url, Kalman)

            #call optimization function
            Init_args=np.array([0.]*len(self.bmk))
            self.opt=self.OptimizationCell(Init_args)
            if self.opt.success:
                if i==0:
                    self.stgy_Wgt=pd.DataFrame()
                    self.stgy_Wgt['Company']=self.etf_dy.index
                    self.stgy_Wgt['Datetime']=date
                    self.stgy_Wgt['% Wgt']=np.round(self.opt.x, 2)+self.etf_dy['% Wgt'].values

                else:
                    temp=pd.DataFrame()
                    temp['Company']=self.etf_dy.index
                    temp['Datetime']=date
                    temp['% Wgt']=np.round(self.opt.x, 2)+self.etf_dy['% Wgt'].values
                    self.stgy_Wgt=pd.concat([self.stgy_Wgt, temp], ignore_index=True)
            else:
                print(self.opt.message)
                break
            print("{} COMPLETED!".format(date))
        self.stgy_Wgt.set_index('Datetime', inplace=True)
        self.stgy_Wgt.to_excel('./data/pca/pca_stgy.xlsx')

if __name__=="__main__":
    dir_path="./data/pca"
    etf_url='./data/pca/pca_etf.xlsx'
    order=1
    state=2
    tns=63
    dval=0.0001
    diff=10
    freq=10
    window=252
    lamb=1
    thresh=0.5
    bt=BackTCell(dir_path, state, order, tns, dval, diff, freq, window,lamb, thresh)
    bt.Strgy(etf_url,True)




