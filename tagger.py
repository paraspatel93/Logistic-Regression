# -*- coding: utf-8 -*-
"""
Predicting Label using Logistic regression.
Created on Tue Feb 20 17:16:03 2018
@author: Paras
"""
from collections import defaultdict
import numpy as np
import time
import sys


np.set_printoptions(threshold=np.nan)
mdl=int(sys.argv[8])

#Model 1
if(mdl==1):
    trainthis=sys.argv[1]
    validatethis=sys.argv[2]
    testthis=sys.argv[3]
    metricthis=sys.argv[6]
    trainout=sys.argv[4]
    testout=sys.argv[5]
    ep=int(sys.argv[7])
    
    file_in=open(trainthis,"r")
    start=time.time()
    with open(trainthis,'r') as file: Xtrain=[Xtrain.strip().split('\t') for Xtrain in file]
    with open(testthis,'r') as file: Xtest=[Xtest.strip().split('\t') for Xtest in file]
    
    data_in=np.genfromtxt(trainthis,dtype=None, delimiter="\t")
    data_test=np.genfromtxt(testthis,dtype=None, delimiter="\t")
    data_validate=np.genfromtxt(validatethis,dtype=None, delimiter="\t")
    
    train_out=open(trainout,"w")
    test_out=open(testout,"w")
    metric=open(metricthis,"w")
    
    epoch=ep
    model=mdl
    col1=data_in[:,0]       #Examples
    col2=Xtrain
    col1_test=data_test[:,0]
    col1_validate=data_validate[:,0]
    unique_label=(np.unique(data_in[:,-1]))
    label=data_in[:,-1]
    label_validate=data_validate[:,-1]
    label_xtrain=[]         #Labels
    label_test=data_test[:,-1]
    features = np.unique(data_in[:,0])  #Unique examples/features
    K=len(np.unique(data_in[:,-1])) #Length of unique labels
    
    
    for i in range(len(Xtrain)):
        if(Xtrain[i]==['']):
            label_xtrain.append(Xtrain[i])
        else:
            label_xtrain.append(Xtrain[i][1])
    col1_xtrain=[]
    for i in range(len(Xtrain)):
        if(Xtrain[i]==['']):
            col1_xtrain.append(Xtrain[i])
        else:
            col1_xtrain.append(Xtrain[i][0])
    
    example_index=defaultdict(list)
    example_index1=defaultdict(list)
    label_index=defaultdict(list)       ## Label to index
    index_label=defaultdict(list)       ##Index to Label
    
    for i in range(0,len(np.unique(data_in[:,0]))):
        index_label[i].append(i)
        
    for i in range(0,len(np.unique(data_in[:,0]))):
        example_index[features[i]].append(i)     
        
    for i in range(0,len(np.unique(data_in[:,0]))):
        example_index1[features[i]].append(i)
    
    for i in range(0,len(np.unique(data_in[:,-1]))):
        label_index[i].append(unique_label[i])
        
    
    def indicator(y,k):
        I=0
        if (y==k):
            I=1
            return I
        else:
            return I
    
    def numerator(exampl,thetaa,c):
        sumn=0
        q=example_index[exampl][0]
        sumn=thetaa[q,c]+thetaa[-1,c]
        return np.exp(sumn)
    
    def denomenator(examp,theta2):
        sumd=0
        indu=example_index[examp][0]
        for l in range(0,len(unique_label)):
            sumd= sumd+ np.exp(theta2[indu][l]+theta2[-1][l])
        return sumd
    
    def prediction(theta,colmn):
        label_predict=[]
        for i in range(len(colmn)):
            a=np.zeros(len(unique_label))
            ex=colmn[i]
            index=example_index[ex][0]
            labelind=0
            for k in range(len(unique_label)):
                q=theta[-1,k]
                w=theta[index,k]
                a[k]=(q+w)
            labelind=np.argmax(a)
            label_predict.append(unique_label[labelind])
            #print(label_predict)
        return label_predict
        
    def error(labels,label_predict):
        true=0
        false=0.0
        for i in range(len(label_predict)):
            if(labels[i]==label_predict[i]):
                true=true+1
            else:
                false=false+1
        return (false/(false+true))   
                
    def likelihood(theta_l,colmn):
        l=0
        for i in range(len(colmn)):
            exa=colmn[i]
            inde=example_index[exa][0]
            deno=denomenator(exa,theta_l)
            for k in range(len(unique_label)):
                indi=indicator(label[i],unique_label[k])
                if(indi==1):
                    nume=np.exp(theta_l[inde,k]+theta_l[-1,k])
                    l+=(indi*np.log(nume/deno))
        return (-l/len(colmn))
    def likelihood_val(theta_l,colmn):
        l=0
        for i in range(len(colmn)):
            exa=colmn[i]
            inde=example_index[exa][0]
            deno=denomenator(exa,theta_l)
            for k in range(len(unique_label)):
                indi=indicator(label_validate[i],unique_label[k])
                if(indi==1):
                    nume=np.exp(theta_l[inde,k]+theta_l[-1,k])
                    l+=(indi*np.log(nume/deno))
        return (-l/len(colmn))
    
    theta=np.transpose(np.zeros((len(unique_label),len(features)+1)))
    theta1=np.transpose(np.zeros((len(unique_label),len(features)+1)))
    grad=np.transpose(np.zeros((len(unique_label),len(features)+1)))
    like_train_plot=[]
    like_validate_plot=[]
    start=time.time()
    epoch_count=0
    while(epoch_count!=epoch): 
        for i in range(len(col1)):  
            ex=col1[i]
            deno = 0
            index=example_index[ex][0]
            deno=denomenator(ex,theta)
            for k in range(len(unique_label)):
                indi=indicator(label[i],unique_label[k])
                nume=np.exp(theta[index,k]+theta[-1,k])
                grad[index][k]=-(indi-(nume/deno))
                grad[-1][k]=-(indi-(nume/deno))
            theta[index]=theta[index]-(0.5*grad[index])
            theta[-1]=theta[-1]-(0.5*grad[-1])
        like_train=likelihood(theta,col1)
        like_validate=likelihood_val(theta,col1_validate)
        #print('epoch=',epoch_count+1,'likelihood(train):',like_train)
        #print('epoch=',epoch_count+1,'likelihood(validation):',like_validate)
        like_train_plot.append(like_train)
        like_validate_plot.append(like_validate)
        metric.write("epoch={0} likelihood(train): {1:.6f}\n".format(epoch_count+1,like_train))
        metric.write("epoch={0} likelihood(validation): {1:.6f}\n".format(epoch_count+1,like_validate))
        epoch_count=epoch_count+1

    train_predict=prediction(theta,col1) 
    test_predict=np.array(prediction(theta,col1_test))
    like_test=likelihood(theta,col1_test)  
    err_train=error(label,train_predict)
    err_test=error(label_test,test_predict)   
    print('error(train):',err_train)
    print('error(test):',err_test)
    metric.write("error(train): %1.6f\n" % err_train)
    metric.write("error(test): %1.6f" % err_test)
    count=0
    for i in (range(len(Xtrain))): 
        if(Xtrain[i]!=['']):
            train_out.write("%s\n" % train_predict[i-count])
        else:
            count+=1
            train_out.write("\n")
    count=0
    for i in (range(len(Xtest))):
        if(Xtest[i]!=['']):
            test_out.write("%s\n" % test_predict[i-count])
        else:
            count+=1
            test_out.write("\n")
        end=time.time()

    train_out.write('\n')
    test_out.write('\n')
    train_out.close()
    file_in.close()
    test_out.close()
    metric.close()
    
#Model 2
if(mdl==2):
    file_train=sys.argv[1]
    file_test=sys.argv[3]
    file_validate=sys.argv[2]
    metricthis=sys.argv[6]
    trainout=sys.argv[4]
    testout=sys.argv[5]
    train_out=open(trainout,"w")
    test_out=open(testout,"w")
    metric=open(metricthis,"w")
    with open(file_train,'r') as file: Xtrain=[Xtrain.strip().split('\t') for Xtrain in file]
    with open(file_test,'r') as file: Xtest=[Xtest.strip().split('\t') for Xtest in file]
    with open(file_validate,'r') as file: Xvali=[Xvali.strip().split('\t') for Xvali in file]
    
    train_data=[]
    train_data.insert(0,'BOS')
    for i in range(len(Xtrain)):
        if(Xtrain[i]==['']):
            train_data.append('EOS');
            train_data.append('BOS');
        else:
            train_data.append(Xtrain[i][0])
    train_data.append('EOS')
    
    validate_data=[]
    validate_data.insert(0,'BOS')
    for i in range(len(Xvali)):
        if(Xvali[i]==['']):
            validate_data.append('EOS');
            validate_data.append('BOS');
        else:
            validate_data.append(Xvali[i][0])
    validate_data.append('EOS')
    test_data=[]
    test_data.insert(0,'BOS')
    for i in range(len(Xtest)):
        if(Xtest[i]==['']):
            test_data.append('EOS');
            test_data.append('BOS');
        else:
            test_data.append(Xtest[i][0])
    test_data.append('EOS')
    
    label_train=[]
    label_train.insert(0,'BOS')
    for i in range(len(Xtrain)):
        if(Xtrain[i]==['']):
            label_train.append('EOS');
            label_train.append('BOS');
        else:
            label_train.append(Xtrain[i][1])
    label_train.append('EOS')
    label_test=[]
    label_test.insert(0,'BOS')
    for i in range(len(Xtest)):
        if(Xtest[i]==['']):
            label_test.append('EOS');
            label_test.append('BOS');
        else:
            label_test.append(Xtest[i][1])
    label_test.append('EOS')
    label_validate=[]
    label_validate.insert(0,'BOS')
    for i in range(len(Xvali)):
        if(Xvali[i]==['']):
            label_validate.append('EOS');
            label_validate.append('BOS');
        else:
            label_validate.append(Xvali[i][1])
    label_validate.append('EOS')
    features=np.unique(train_data)
    K_temp=np.unique(label_train)
    K=[]
    for i in range(len(K_temp)):
        if(K_temp[i]!='BOS' and K_temp[i]!='EOS'):
            K.append(K_temp[i])
    feature_dict=defaultdict(list)
    label_dict=defaultdict(list)
    for i in range(len(features)):
        feature_dict[features[i]].append(i)
    for i in range(len(K)):
        label_dict[K[i]].append(i)
    epoch=int(sys.argv[7])
    epoch_count=0
    theta=(np.zeros(((len(features)*3)+1,len(K))))
    grad=(np.zeros(((len(features)*3)+1,len(K))))
    
    def den(theta1,curr,prevv,nexttt):
        sumd=0
        for i in range(len(K)):
            sumd+=np.exp(theta1[curr][i]+theta1[prevv][i]+theta1[nexttt][i]+theta1[-1][i])
        return sumd
    def indicator(label1,label2):
        I=0
        if(label1==label2):
            I=1
            return I
        else:
            return I
    def likelihood2(thet,colmn,lab):
        J=0
        count=0
        for i in range(len(colmn)):
            if(colmn[i]=='BOS' or colmn[i]=='EOS'):
                continue
            else:
                count+=1
                curr=feature_dict[colmn[i]][0]+(len(features))
                prevv=feature_dict[colmn[i-1]][0]
                nexttt=(feature_dict[colmn[i+1]][0])+(2*len(features))
                deno=den(thet,curr,prevv,nexttt)
                for j in range(len(lab)):
                    nume=np.exp(thet[curr][j]+thet[prevv][j]+thet[nexttt][j]+thet[-1][j])
                    indi=indicator(K[j],label_train[i])
                    J+=indi*np.log(nume/deno)
        return (-J/count) 
    def likelihood2_val(thet,colmn,lab):
        J=0
        count=0
        for i in range(len(colmn)):
            if(colmn[i]=='BOS' or colmn[i]=='EOS'):
                continue
            else:
                count+=1
                curr=feature_dict[colmn[i]][0]+(len(features))
                prevv=feature_dict[colmn[i-1]][0]
                nexttt=(feature_dict[colmn[i+1]][0])+(2*len(features))
                deno=den(thet,curr,prevv,nexttt)
                for j in range(len(lab)):
                    nume=np.exp(thet[curr][j]+thet[prevv][j]+thet[nexttt][j]+thet[-1][j])
                    indi=indicator(K[j],label_validate[i])
                    J+=indi*np.log(nume/deno)
        return (-J/count)
        
    def prediction2(theta,train_data):
        label_predict=[]
        
        for i,example in  enumerate(train_data):
            a=[]
            if(example=='BOS' or example=='EOS'):
                label_predict.append('BOS')
                continue
            else:
                cur=feature_dict[train_data[i]][0]+(len(features))
                prev=feature_dict[train_data[i-1]][0]
                nextt=(feature_dict[train_data[i+1]][0])+(2*len(features))
                for j in range(len(K)):
                    num=np.exp(theta[cur][j]+theta[prev][j]+theta[nextt][j]+theta[-1][j])
                    a.append(num)
                a_max=np.argmax(a)
            label_predict.append(K[a_max])
        return label_predict
    def error2(label_train1,label_pre):
        true=0
        false=0.0
        for i in range(len(label_pre)):
            if(label_pre[i]=='BOS'):
                continue
            elif(label_pre[i]==label_train1[i]):
                true+=1
            else:
                false+=1
        return (false/(false+true))
    while(epoch_count!=epoch):
        for i in  range(len(train_data)):
            example=train_data[i]
            if(example=='BOS' or example=='EOS'):
                continue
            else:
                cur=feature_dict[train_data[i]][0]+(len(features))
                prev=feature_dict[train_data[i-1]][0]
                nextt=(feature_dict[train_data[i+1]][0])+(2*len(features))
                deno=den(theta,cur,prev,nextt)

                for j in range(len(K)):
                    nume=np.exp(theta[cur][j]+theta[prev][j]+theta[nextt][j]+theta[-1][j])
                    indi=indicator(K[j],label_train[i])
                    grad[cur][j]=-(indi-(nume/deno))
                    grad[nextt][j]=-(indi-(nume/deno))
                    grad[prev][j]=-(indi-(nume/deno))
                    grad[-1][j]=-(indi-(nume/deno))
                    #print(grad[cur])
                    
                theta[cur]=theta[cur]-(0.5*grad[cur])
                theta[nextt]=theta[nextt]-(0.5*grad[nextt])
                theta[prev]=theta[prev]-(0.5*grad[prev])
                theta[-1]=theta[-1]-(0.5*grad[-1])
        like_train=likelihood2(theta,train_data,K)
        like_validate=likelihood2_val(theta,validate_data,K)
        metric.write("epoch={0} likelihood(train): {1:.6f}\n".format(epoch_count+1,like_train))
        metric.write("epoch={0} likelihood(validation): {1:.6f}\n".format(epoch_count+1,like_validate))
        epoch_count+=1
    Z_train=prediction2(theta,train_data)
    Z_test=prediction2(theta,test_data)
    err_train=error2(label_train,Z_train)
    err_test=error2(label_test,Z_test)
    print('error(train):',err_train)
    print('error(test):',err_test)
    metric.write("error(train): %1.6f\n" % err_train)
    metric.write("error(test): %1.6f" % err_test)
    for i in (range(len(train_data))):
        if(train_data[i]=='BOS'): 
            continue
        elif(train_data[i]=='EOS'):
            train_out.write("\n")
        else:
            train_out.write("%s\n" % Z_train[i])
    for i in (range(len(test_data))):
        if(test_data[i]=='BOS'): 
            continue
        elif(test_data[i]=='EOS'):
            test_out.write("\n")
        else:
            test_out.write("%s\n" % Z_test[i])
    metric.close()
    test_out.close()
    train_out.close()