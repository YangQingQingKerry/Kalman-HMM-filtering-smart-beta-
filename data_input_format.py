# -*- coding: utf-8 -*-
#pure python programming

import os
import pandas as pd
from pandas import ExcelWriter

path=os.getcwd()
print("The current working directory is %s"%path)

folder=path+'/data/pca'
try:
    os.mkdir(folder)
except OSError:
    print("%s has been created" %folder)
else:
    print("Successfully created the directory %s"%folder)


def etfExcel():
    df=pd.ExcelFile('./data/raw/etf.xls')
    sheets=df.sheet_names
    etf=df.parse(sheets[0], skiprows=9, usecols="C,E:I")
    etf.columns=['Company',
                 '% Wgt',
                 'Market Value',
                 'Position',
                 'Closing Price',
                 'Currency']
    etf=etf[~etf['Closing Price'].isna()]
    etf['Company']=etf['Company'].apply(lambda x: x.capitalize())

    for i in range(len(sheets)-1):
        t=df.parse(sheets[i+1], skiprows=9, usecols="C, E:I")
        t.columns=['Company',
                 '% Wgt',
                 'Market Value',
                 'Position',
                 'Closing Price',
                 'Currency']
        t=t[~t['Closing Price'].isna()]
        t['Company']=t['Company'].apply(lambda x:x.capitalize())
        etf=pd.concat([etf, t], ignore_index=True)

    writer=ExcelWriter(folder+'/pca_etf.xlsx')
    etf.to_excel(writer, 'hist')
    writer.save()


def indexExcel():
    df=pd.ExcelFile('./data/raw/tsx60.xls')
    sheets=df.sheet_names
    Map=dict()
    for tp in zip(df.parse(sheets[0], skiprows=1)['Ticker'].values, df.parse(sheets[0], skiprows=1)['Name'].values):
        Map[tp[0].replace('/',' ')]=tp[1].capitalize()

    tsx60=df.parse(sheets[1], skiprows=0, usecols='A:B')
    tsx60.columns=['Date', Map[sheets[1]]]


    '''
    (empty sheets:100,101)
    1671922D CN equity -- L.TO
    1410329D CN Equity -- PSK.TO
    '''
    for i in range(98):
        t=df.parse(sheets[i+2], shiprows=0, usecols='A:B')
        t.columns=['Date', Map[sheets[i+2]]]
        tsx60=pd.merge(tsx60, t, how='outer', on='Date')

    tsx60['Date']=pd.to_datetime(tsx60.Date)


    '''
    fetch data for the empty sheets from yahoo finance
    '''
    l_df=pd.read_csv("./data/yahoo/L.TO.csv")[['Date','Adj Close']].copy()
    l_df.columns=['Date', 'Loblaw cos ltd']
    l_df['Date']=pd.to_datetime(l_df.Date)
    tsx60=pd.merge(tsx60, l_df, how='outer', on='Date')
    psk_df=pd.read_csv("./data/yahoo/PSK.TO.csv")[['Date','Adj Close']].copy()
    psk_df.columns=['Date', 'Prairiesky royalty ltd']
    psk_df['Date']=pd.to_datetime(psk_df.Date)
    tsx60=pd.merge(tsx60, psk_df, how='outer', on='Date')


    for i in range(102, 104):
        t=df.parse(sheets[i], shiprows=0, usecols='A:B')
        t.columns=['Date', Map[sheets[i]]]
        t['Date']=pd.to_datetime(t.Date)
        tsx60=pd.merge(tsx60, t, how='outer', on='Date')

    writer = ExcelWriter(folder+'/pca_tsx60.xlsx')
    tsx60.to_excel(writer,'hist')
    writer.save()






