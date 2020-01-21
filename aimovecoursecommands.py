# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:48:17 2019

@author: aimove
"""
from hmmlearn.hmm import GaussianHMM 
import numpy as np
import glob
from aimovecoursestatespace import state_space,params_coeff
from sklearn.metrics import confusion_matrix

#paths for data fetching
path1='gesture_commands/Hello/*.txt'
path2='gesture_commands/Left/*.txt' 
path3='gesture_commands/Right/*.txt'
path4='gesture_commands/speed_up/*.txt'
path5='gesture_commands/speed_down/*.txt'

#paths of the files to train the State-Space
path6='gesture_commands/Hello/11.txt'
path7='gesture_commands/Left/14.txt' 
path8='gesture_commands/Right/13.txt'
path9='gesture_commands/speed_up/13.txt'
path10 ='gesture_commands/speed_down/13.txt'

nopeCount=0
ss_score=0
ss_hmm_score_counter =0
score=0
train_num=0.8
rms_counter_hello=0
rms_counter_left=0
rms_counter_right=0
rms_counter_speed_up=0
rms_counter_speed_down=0

dimensions=14
y_pred_hmm=[]
y_pred_ss=[]
y_pred_ss_hmm=[]
y_true=[]
usedJoints=[0,1,3,4,6,7,9,10,12,13,15,16,18,19] #7 joints
#usedJoints=[9,10,18,19] #2 joints

#counter to take each time a different data txt
testSampleGesture2=-1

#lists to take each gesture
file_list1 = glob.glob(path1)
file_list2 = glob.glob(path2)
file_list3 = glob.glob(path3)
file_list4 = glob.glob(path4)
file_list5 = glob.glob(path5)

#merge lists
file_all=file_list1+file_list2+file_list3+file_list4+file_list5

#lists for the parameters coefficients of State Space
psList6=params_coeff(path6)
psList7=params_coeff(path7)
psList8=params_coeff(path8)
psList9=params_coeff(path9)
psList10=params_coeff(path10)

#testSampleGesture --> the classes, testSample --> iterations per gesture
for testSampleGesture in range(5):
    for testSample in range(0,16):
        testSampleGesture2+=1
        
        states=8 #number of hidden states
        
        file_list1 = glob.glob(path1)
        file_list2 = glob.glob(path2)
        file_list3 = glob.glob(path3)
        file_list4 = glob.glob(path4)
        file_list5 = glob.glob(path5)

        #from file_all data take iterations one by one
        test_data=file_all[testSampleGesture2]
        
        train_data1=file_list1
        train_data2=file_list2
        train_data3=file_list3
        train_data4=file_list4
        train_data5=file_list5
        
        # test the class we are in and pop the iteration from the respective train_data list
        if (testSampleGesture==0):
            train_data1.pop(testSample)
        elif (testSampleGesture==1):
            train_data2.pop(testSample)
        elif (testSampleGesture==2):
            train_data3.pop(testSample)    
        elif (testSampleGesture==3):
            train_data4.pop(testSample)
        elif (testSampleGesture==4):
            train_data5.pop(testSample)
        
        #make lists with zeros with dimensions 1x14 (7 joints*2 coordinates=14)
        x1=np.zeros((1,dimensions))
        x2=np.zeros((1,dimensions))
        x3=np.zeros((1,dimensions))
        x4=np.zeros((1,dimensions))
        x5=np.zeros((1,dimensions))
        
        #make lists that will keep the length of x1,x2,x3,x4,x5
        x1_lengths=[]
        x2_lengths=[]
        x3_lengths=[]
        x4_lengths=[]
        x5_lengths=[]
        
        #loop for loading the txt files one by one of each training set
        #We take all iterations from the same gesture & add all of them in an array and distinguish them through the length of each iteration
        for file in train_data1:
            data=np.loadtxt(file)[:,usedJoints]-100 #-100 is used to bring the data in the format that we need them
            x1 = np.concatenate((x1, data), axis=0) 
            x1_lengths.append(data.shape[0]) #list with the length of each time-series
        
        for file in train_data2:
            data=np.loadtxt(file)[:,usedJoints]-100
            x2 = np.concatenate((x2, data), axis=0)
            x2_lengths.append(data.shape[0]) 
            
        for file in train_data3:
            data=np.loadtxt(file)[:,usedJoints]-100
            x3 = np.concatenate((x3, data), axis=0)
            x3_lengths.append(data.shape[0]) 
           
        
        for file in train_data4:
            data=np.loadtxt(file)[:,usedJoints]-100
            x4 = np.concatenate((x4, data), axis=0)
            x4_lengths.append(data.shape[0]) 
            
        for file in train_data5:
            data=np.loadtxt(file)[:,usedJoints]-100
            x5 = np.concatenate((x5, data), axis=0)
            x5_lengths.append(data.shape[0]) 

        #train Gaussian HMMs & define parameters for each gesture
        model1=GaussianHMM(n_components=1, covariance_type='full',verbose=False).fit(x1[1:],x1_lengths) #4
        model2=GaussianHMM(n_components=3,covariance_type='diag',verbose=False).fit(x2[1:],x2_lengths) #3
        model3=GaussianHMM(n_components=4,covariance_type='diag',verbose=False).fit(x3[1:],x3_lengths) #4
        model4=GaussianHMM(n_components=1, covariance_type='full',verbose=False).fit(x4[1:],x4_lengths) #4
        model5=GaussianHMM(n_components=1,covariance_type='full',verbose=False).fit(x5[1:],x5_lengths) #9
        
        #load test data
        data=np.loadtxt(test_data)[:,usedJoints]-100
        #list to store HMM results
        scoreList=[]
        #HMMs test 
        score_mode1=model1.score(data)
        score_mode2=model2.score(data)
        score_mode3=model3.score(data)
        score_mode4=model4.score(data)
        score_mode5=model5.score(data)
        scoreList=[score_mode1,score_mode2,score_mode3,score_mode4, score_mode5]
        
        #find the maximum log-likelihood from the list with all the HMM results
        winner_prob=max(scoreList)
        index= scoreList.index(max(scoreList))
        
        #conversion of a list to array
        scoreList=np.asarray(scoreList)
        
        #conversion of log-likelihood to probability
        score_to_percent=np.interp(scoreList, (scoreList.min(), scoreList.max()), (0, +1))
        
        #find the winner class
        if(winner_prob==score_mode1):
            winner_model='Hello'
        elif(winner_prob==score_mode2):
            winner_model='Left'
        elif(winner_prob==score_mode3):
            winner_model='Right'
        elif(winner_prob==score_mode4):
            winner_model='speed_up'  
        else:
            winner_model='speed_down'
            
        #test if probability is smaller than 55 (0.55) & if yes compute the results of the State-Space method
        if (100*score_to_percent[index]/np.sum(score_to_percent))<55:
            
            score_g1=state_space(path6,test_data,psList6)
            score_g2=state_space(path7,test_data,psList7)
            score_g3=state_space(path8,test_data,psList8)
            score_g4=state_space(path9,test_data,psList9)
            score_g5=state_space(path10,test_data,psList10)
          
            #list with the State-Space results
            ssScoreList=[score_g1,score_g2,score_g3,score_g4,score_g5]
            winner=max(ssScoreList)
            
            ss_index=ssScoreList.index(max(ssScoreList))
            ssScoreList=np.asarray(ssScoreList).reshape(5)
        
            ss_score_to_percent=np.interp(ssScoreList, (ssScoreList.min(), ssScoreList.max()), (0, +1))
            
            #computation of combined result (State-Space & HMMs)
            ss_hmm_score=ssScoreList*100*score_to_percent/np.sum(score_to_percent)
            
            if winner==score_g1:
                state_space_pred='Hello'
            if winner==score_g2:
                state_space_pred='Left'   
            if winner==score_g3:
                state_space_pred='Right'
            if winner==score_g4:
                state_space_pred='speed_up'
            if winner==score_g5:
                state_space_pred='speed_down'
        else:
            state_space_pred='nope' #getting this message when State-Space is not used for the computation of the final result
            nopeCount+=1
            ssScoreList=[0,0,0,0]
            ss_hmm_score=100*score_to_percent/np.sum(score_to_percent)
        ss_hmm_score=list(ss_hmm_score)
            
        ss_hmm_index= ss_hmm_score.index(max(ss_hmm_score))
        
        if ss_hmm_index==0:
            ss_hmm_pred='Hello'
        if ss_hmm_index==1:
            ss_hmm_pred='Left'   
        if ss_hmm_index==2:
            ss_hmm_pred='Right'
        if ss_hmm_index==3:
            ss_hmm_pred='speed_up'
        if ss_hmm_index==4:
            ss_hmm_pred='speed_down'    
        if(test_data.__contains__(winner_model)):
            score+=1  
            print('###########################################################')
            print(test_data)
            print('hmm = '+ winner_model)
            print('ss = '+ state_space_pred)
            print('ss + hmm= '+ ss_hmm_pred)
            print('hmm = '+ str(np.around(100*score_to_percent/np.sum(score_to_percent),decimals=3)))
            print('ss = '+ str(np.around(ssScoreList,decimals=3)))        
            print('ss + hmm= '+ str(np.around(ss_hmm_score,decimals=3)))
        else:
            print('###########################################################')
            print(test_data)
            print('hmm = '+ winner_model+'!!!!!!')
            print('ss = '+ state_space_pred)
            print('ss + hmm= '+ ss_hmm_pred)
            print('hmm = '+ str(np.around(100*score_to_percent/np.sum(score_to_percent),decimals=3)))
            print('ss = '+ str(np.around(ssScoreList,decimals=3)))        
            print('ss + hmm= '+ str(np.around(ss_hmm_score,decimals=3)))
            
        if(test_data.__contains__(ss_hmm_pred)):
            ss_hmm_score_counter+=1
        y_pred_hmm.append(winner_model)
        y_pred_ss.append(state_space_pred)
        y_pred_ss_hmm.append(ss_hmm_pred)
        
        if(test_data.__contains__(state_space_pred)):
            ss_score+=1
        
        if(test_data.__contains__('Hello')):
            y_true.append('Hello')
        elif(test_data.__contains__('Left')):
            y_true.append('Left')
        elif(test_data.__contains__('Right')):
            y_true.append('Right')
        elif(test_data.__contains__('speed_up')):
            y_true.append('speed_up')
        else:
            y_true.append('speed_down')   

#compute & print results        
confusion_hmm= confusion_matrix(y_true, y_pred_hmm, labels=["Hello", "Left", "Right", "speed_up","speed_down" ])
confusion_ss= confusion_matrix(y_true, y_pred_ss, labels=["Hello", "Left", "Right", "speed_up","speed_down" ])
confusion_ss_hmm= confusion_matrix(y_true, y_pred_ss_hmm, labels=["Hello", "Left", "Right", "speed_up","speed_down" ])

print('#############  HMM  #############')
print(confusion_hmm) 
print('#############  SS  #############')
print(confusion_ss)  
print('#############  SS and HMM  #############')
print(confusion_ss_hmm)     
print('hmm total score is:'+str(score)+'/'+str(80)) 
print('ss total score is:'+str(ss_score)+'/'+str(80-nopeCount)) 
print('ss+hmm total score is:'+str(ss_hmm_score_counter)+'/'+str(80))            
lengthtest = len(test_data)
percentage = ((score*100)/80)
print('percentage is:' +str(percentage) +'%')  
  