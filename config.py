#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class Config:
    def __init__(self, f, state, order, tns):
        self.k=len(self.f.columns)
        self.state=state
        self.order=order
        self.tns=tns
        self.d=f.iloc[:self.tns].copy()

    def conFig(self):
        df=self.d.copy()
        df_1=df.shift(1).fillna(0)
        #-----
        T=df.sum()
        T1=df_1.sum()
        TT1=(df*df_1).sum()
        T1_sq=(df_1**2).sum()
        #-----
        a1=self.tns*TT1-T*T1
        b11=self.tns*T1_sq-T1**2
        if self.order==1:
            MT1=[1.0]
            MT2=[0.5,0.5,0.5,0.5]
            MT3=[1/3]*9
        else: #2-order
            MT1=[1.]
            MT2=[0.5,0.5,0,0,
                 0,0,0.5,0.5,
                 0.5,0.5,0,0,
                 0,0,0.5,0.5]
            MT3=[1/3,1/3,1/3,0,0,0,0,0,0,
                 0,0,0,1/3,1/3,1/3,0,0,0,
                 0,0,0,0,0,0,1/3,1/3,1/3,
                 1/3,1/3,1/3,0,0,0,0,0,0,
                 0,0,0,1/3,1/3,1/3,0,0,0,
                 0,0,0,0,0,0,1/3,1/3,1/3,
                 1/3,1/3,1/3,0,0,0,0,0,0,
                 0,0,0,1/3,1/3,1/3,0,0,0,
                 0,0,0,0,0,0,1/3,1/3,1/3]
        theta=a1/b11
        mu=(T-T1*theta)/self.tns
        rm=df-theta*df_1-mu
        sigma2=(rm**2).sum()/self.tns
        if self.state==1:
            self.args=MT1+list(theta)+list(mu)+list(sigma2)
            self.args=np.array(self.args)
        elif self.state==2:
            self.args=MT2+list(theta)*2+list(mu)\
                      +list(mu+ np.array([0.0001]*self.k))+list(sigma2)*2
            self.args=np.array(self.args)
        else: #state==3
            self.args=MT3+list(theta)*3+list(mu)*2\
                      +list(mu+ np.array([0.0001]*self.k))+list(sigma2)*3
            self.args=np.array(self.args)









