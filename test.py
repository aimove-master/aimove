# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:48:17 2019

@author: aimove
"""

"""
Code to find for each hidden state of the HMMs, the iteration that gives the highest log-likelihood
The results are 1. the number of the states 2.name of gesture with the highest log-likelihood result
3.the highest log-likelihood
"""
from hmmlearn.hmm import GaussianHMM 
import numpy as np
import glob

path1='gesture_commands/Hello/*.txt'
path2='gesture_commands/Left/*.txt'
path3='gesture_commands/Right/*.txt'
path4='gesture_commands/speed_up/*.txt'
path5='gesture_commands/speed_down/*.txt'


stateList=[]
winnerList=[]
scoreList=[]

for state_N in range(1,25):
    
    scoreList2=[]
    ss_score=0
    ss_hmm_score_counter=0
    score=0
    train_num=0.8
    dimensions=4
    y_pred=[]
    y_true=[]
    usedJoints=[9,10,18,19]
    #usedJoints=[0,1,3,4,6,7,9,10,12,13,15,16,18,19] #7 joints
    states=state_N #number of hidden states
    
    x1=np.zeros((1,dimensions))
    
    
    x1_lengths=[]
    
    test_data=sorted(glob.glob(path1)) #!!!!!!!!!! select path !!!!!!!!!
    
    for i in range(16):

        
        data=np.loadtxt(test_data[i],delimiter=' ')[:,usedJoints]-100

        x1_lengths=[]
        x1_lengths.append(data.shape[0])
        #different topology results to different results and needs different states configuration
        model1=GaussianHMM(n_components=states, covariance_type='diag',).fit(data,x1_lengths)

        
        score=0
        #loop for finding mean of the log-likelihoods in each iteration 
        for file in test_data:
            data2=np.loadtxt(file,delimiter=' ')[:,usedJoints]-100
            score_mode1=model1.score(data2)
            score+=score_mode1/16
        scoreList2.append(score)    
    winner_prob=max(scoreList2)
    index= scoreList2.index(max(scoreList2))
    best_iter=test_data[index]
    
    stateList.append(states)
    winnerList.append(best_iter)
    scoreList.append(winner_prob)
    
for i in range(len(stateList)):
    print(str(stateList[i])+' '+str(winnerList[i])+' '+str(int(scoreList[i])))
 