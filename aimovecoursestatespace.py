# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:45:22 2019

@author: aimove
"""
# Load the statsmodels api
import statsmodels.api as sm
import numpy as np

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import warnings
warnings.filterwarnings("ignore")

# function for picking endogenous & exogenous variables for the State-Space model 
def params_coeff(path1):
    psList=[]
    crop=0
    for coor in [0,1,3,4]:

        joint=coor
        order=2
    
        data_g1=np.loadtxt(path1)[:,[9,10,11,18,19,20]]-100
        
        data_g1=data_g1[crop:data_g1.shape[0]-crop,:]

        endog=data_g1[:,joint]
        jointList=[]
        
        if joint<=2:
        
            jointList=[0,1]
            jointList.pop(joint)    
            exog = data_g1[:,[jointList[0],joint+3]]
        else:
            jointList=[3,4]
            jointList.pop(joint-3)    
            exog = data_g1[:,[jointList[0],joint-3]]
            
        # We could fit an AR(2) model, described above
        res_sarimax = sm.tsa.SARIMAX(endog,exog=exog, order=(order,0,0 )).fit(solver='bfgs')
             
        # Show the summary of results
        #print(res_sarimax.summary())
        
        ps=res_sarimax.params
        psList.append(ps)  
        
    return psList
    
#implementation of the State-Space method
def state_space(path1,path2,psList):

    crop=0 #to crop frames, in this case the first and last ones
    score_coor=[]
    score_coor2=[]
    
    ps_i=0
    
    for coor in [0,1,3,4]:
        
        ps=psList[ps_i]
        ps_i+=1

        joint=coor
    
        data_g1=np.loadtxt(path1)[:,[9,10,11,18,19,20]]-100
        data_g1[crop:data_g1.shape[0]-crop,:]
        
        data2_g1=np.loadtxt(path2)[:,[9,10,11,18,19,20]]-100
        data2_g1[crop:data2_g1.shape[0]-crop,:]
    
        #use of dtw to allign the time-series
        tupleList=[]
        tupleList2=[]
        
        distance, path = fastdtw(data_g1, data2_g1,radius=(data_g1.shape[0]+data2_g1.shape[0]), dist=euclidean)
        
        for i in path:
            tupleList.append(data_g1[i[0],:])
            tupleList2.append(data2_g1[i[1],:])
 
        data_dtw=np.asarray(tupleList)
        data2_dtw=np.asarray(tupleList2)
        
        data_g1=data_dtw
        data2_g1=data2_dtw
        
        nob=data_g1.shape[0]
        nob2=data2_g1.shape[0]
        endog=data_g1[:,joint]
        jointList=[]
        
        
        if joint<=2:
        
            jointList=[0,1]
            jointList.pop(joint)    
            exog = data_g1[:,[jointList[0],joint+3]]
            exog2=data2_g1[:,[jointList[0],joint+3]]
        else:
            jointList=[3,4]
            jointList.pop(joint-3)    
            exog = data_g1[:,[jointList[0],joint-3]]
            exog2 = data2_g1[:,[jointList[0],joint-3]]

        yList=[0,0] #list for the training forecast
        yDiff=[0,0]
        
        #computation of the forecasting 
        #for training
        B=np.array([ps[0],ps[1]]).reshape(1,2)

        for i in range(1,nob-1):
            a=np.dot(np.array([[ps[2],ps[3]],[1,0]]),np.array([data_g1[i,joint],data_g1[i-1,joint]]))
            a=np.asarray(a)
            y=np.dot(np.array([1,0]).reshape(1,2),a)+np.dot(B,exog[i,:].T)
            yList.append(float(y))
            yDiff.append(data_g1[i+1,joint]-yList[i+1])
        
        #for testing
        yList3=[0,0] #list for testing forecast
        
        for i in range(1,nob2-1):
            
            a=np.dot(np.array([[ps[2],ps[3]],[1,0]]),np.array([data2_g1[i,joint],data2_g1[i-1,joint]]))
            a=np.asarray(a)
            y=np.dot(np.array([1,0]).reshape(1,2),a)+np.dot(B,exog2[i,:].T)
            yList3.append(float(y))
        
        #bringing the forecast close to the original time-series 
        forecast=np.asarray(yList) +float(np.mean(np.asarray(yDiff)))
        forecast2=np.asarray(yList3) +float(np.mean(np.asarray(yDiff)))
        #deleting the offset between the forecasted time-series (training and testing ones)
        forecast2=forecast2-np.mean(forecast2[2:]-forecast[2:])
        forecast=forecast[2:]
        forecast2=forecast2[2:]
       
        diff=abs(endog[2:]-forecast)
       
        #upperBounds= forecast+diff+np.min(diff)
        #lowerBounds =forecast-diff-np.min(diff)    
        
    
        if(coor!=2 and coor!=5):   
                      
            score_coor.append(forecast)
            score_coor2.append(forecast2)

    aa=np.asarray(score_coor2[0])
    aa=aa.reshape(aa.shape[0],1)
    bb=np.asarray(score_coor[0])
    bb=bb.reshape(bb.shape[0],1)
    
    for i in range(1,4):
        aa=np.concatenate((aa,np.asarray(score_coor2[i]).reshape(aa.shape[0],1)),axis=1)
        bb=np.concatenate((bb,np.asarray(score_coor[i]).reshape(bb.shape[0],1)),axis=1)
        
    distance2, pathppp = fastdtw(aa, bb,radius=1, dist=euclidean)

    return 1/(1+distance2)
 
