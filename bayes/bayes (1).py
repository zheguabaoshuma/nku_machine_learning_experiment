import numpy as np
import matplotlib.pyplot as plt
import csv


np.random.seed(25565)
total_samplenum=0
total_testnum=0
sample=np.random.rand(2000,2)*10
features=np.zeros([2000,2])
tag=np.zeros(2000)
#num=len(sample)
test=np.random.rand(100,2)*10
test_features=np.zeros([100,2])
test_tag=np.zeros(100)
noise=np.random.rand(1000,2)*10
noise_features=np.zeros([1000,2])
noise_tag=np.random.randint(1,10,2000)
noise_num=len(noise)
default_color= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
data_row:list=[]

file=open('data.csv','w')
writer=csv.writer(file)

def prior_probability(ntag:int,record:bool=True)->float:
    ntag_num=0
    for i in range(0,len(sample)):
        if tag[i]==ntag:
            ntag_num+=1
    for k in range(0,len(noise)):
        if noise_tag[k]==ntag:
            ntag_num+=1
    result=ntag_num/(total_samplenum+noise_num)
    if record:
        data_row.append(result)
    return result

def conditional_probability(conclusion:int,condition:int,seq:int)->float:
    Ixy:int=0
    Iy=prior_probability(conclusion,False)*(total_samplenum+noise_num)
    #print(Iy)

    for k in range(0,len(sample)):
        if features[k][seq]==condition and tag[k]==conclusion:
            Ixy+=1
    for k in range(0,len(noise)):
        if noise_features[k][seq]==condition and noise_tag[k]==conclusion:
            Ixy+=1
    result=Ixy/Iy
    data_row.append(result)
    return result

def united_probability(x_features:np.array,conclusion:int):
    result=conditional_probability(conclusion,x_features[0],0)*\
           conditional_probability(conclusion,x_features[1],1)*prior_probability(conclusion)
    data_row.append(result)
    return result

def bayes_arg_max(x_features:np.array):
    max_probability:float=0
    max_conclusion:int=0
    for k in range(1,10):
        p=united_probability(x_features,k)
        if p>max_probability:
            max_probability=p
            max_conclusion=k

    data_row.append(max_conclusion)
    data_row.append(max_probability)
    writer.writerow(data_row)
    data_row.clear()
    return max_conclusion
i=0
while i <len(sample):
    if((sample[i][0]>=3 and sample[i][0]<=3.5) or (sample[i][0]>=6.5 and sample[i][0]<=7)
     or (sample[i][1]>=3 and sample[i][1]<=3.5) or (sample[i][1]>=6.5 and sample[i][1]<=7)):
        sample=np.delete(sample,i,0)
        tag=np.delete(tag,i,0)
        features=np.delete(features,i,0)
        #num-=1
        continue
    if(sample[i][0]>=0 and sample[i][0]<=3 and sample[i][1]>=0 and sample[i][1]<=3):
        features[i]=np.array([1,1])
        tag[i]=1
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[0])
    elif(sample[i][0]>=3.5 and sample[i][0]<=6.5 and sample[i][1]>=0 and sample[i][1]<=3):
        features[i]=np.array([2,1])
        tag[i]=2
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[1])
    elif(sample[i][0]>=7 and sample[i][0]<=10 and sample[i][1]>=0 and sample[i][1]<=3):
        features[i]=np.array([3,1])
        tag[i]=3
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[2])
    elif(sample[i][0]>=0 and sample[i][0]<=3 and sample[i][1]>=3.5 and sample[i][1]<=6.5):
        features[i]=np.array([1,2])
        tag[i]=4
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[3])
    elif(sample[i][0]>=3.5 and sample[i][0]<=6.5 and sample[i][1]>=3.5 and sample[i][1]<=6.5):
        features[i]=np.array([2,2])
        tag[i]=5
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[4])
    elif(sample[i][0]>=7 and sample[i][0]<=10 and sample[i][1]>=3.5 and sample[i][1]<=6.5):
        features[i]=np.array([3,2])
        tag[i]=6
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[5])
    elif(sample[i][0]>=0 and sample[i][0]<=3 and sample[i][1]>=7 and sample[i][1]<=10):
        features[i]=np.array([1,3])
        tag[i]=7
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[6])
    elif(sample[i][0]>=3.5 and sample[i][0]<=6.5 and sample[i][1]>=7 and sample[i][1]<=10):
        features[i]=np.array([2,3])
        tag[i]=8
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[7])
    elif(sample[i][0]>=7 and sample[i][0]<=10 and sample[i][1]>=7 and sample[i][1]<=10):
        features[i]=np.array([3,3])
        tag[i]=9
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[8])
    total_samplenum+=1
    i+=1

i=0
while i <len(test):
    if((test[i][0]>=3 and test[i][0]<=3.5) or (test[i][0]>=6.5 and test[i][0]<=7)
     or (test[i][1]>=3 and test[i][1]<=3.5) or (test[i][1]>=6.5 and test[i][1]<=7)):
        test=np.delete(test,i,0)
        test_features=np.delete(test_features,i,0)
        test_tag=np.delete(test_tag,i,0)
        continue
    if(test[i][0]>=0 and test[i][0]<=3 and test[i][1]>=0 and test[i][1]<=3):
        test_features[i]=np.array([1,1])
        test_tag[i]=1
        #plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=3.5 and test[i][0]<=6.5 and test[i][1]>=0 and test[i][1]<=3):
        test_features[i]=np.array([2,1])
        test_tag[i]=2
        #plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=7 and test[i][0]<=10 and test[i][1]>=0 and test[i][1]<=3):
        test_features[i]=np.array([3,1])
        test_tag[i]=3
        #plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=0 and test[i][0]<=3 and test[i][1]>=3.5 and test[i][1]<=6.5):
        test_features[i]=np.array([1,2])
        test_tag[i]=4
        #plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=3.5 and test[i][0]<=6.5 and test[i][1]>=3.5 and test[i][1]<=6.5):
        test_features[i]=np.array([2,2])
        test_tag[i]=5
        #plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=7 and test[i][0]<=10 and test[i][1]>=3.5 and test[i][1]<=6.5):
        test_features[i]=np.array([3,2])
        test_tag[i]=6
        #plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=0 and test[i][0]<=3 and test[i][1]>=7 and test[i][1]<=10):
        test_features[i]=np.array([1,3])
        test_tag[i]=7
        #plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=3.5 and test[i][0]<=6.5 and test[i][1]>=7 and test[i][1]<=10):
        test_features[i]=np.array([2,3])
        test_tag[i]=8
        #plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=7 and test[i][0]<=10 and test[i][1]>=7 and test[i][1]<=10):
        test_features[i]=np.array([3,3])
        test_tag[i]=9
        #plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    total_testnum+=1
    i+=1

i=0
while i <len(noise):
    if((noise[i][0]>=3 and noise[i][0]<=3.5) or (noise[i][0]>=6.5 and noise[i][0]<=7)
     or (noise[i][1]>=3 and noise[i][1]<=3.5) or (noise[i][1]>=6.5 and noise[i][1]<=7)):
        noise=np.delete(noise,i,0)
        noise_tag=np.delete(noise_tag,i,0)
        noise_features=np.delete(noise_features,i,0)
        noise_num-=1
        continue
    else:
        plt.scatter(noise[i][0],noise[i][1],marker='3',c=default_color[int(noise_tag[i])-1])
        noise_features[i]=np.array([int(noise[i][0]/3.5)+1,int(noise[i][1]/3.5)+1])
    i+=1


correct=0
for k in range(0,len(test)):
    pred=bayes_arg_max(test_features[k])
    plt.scatter(test[k][0], test[k][1], marker='s', c=default_color[pred-1], s=15, alpha=0.5)
    if pred==test_tag[k]:
        correct+=1
plt.show()
print("acc is "+str(correct/len(test)))
print("total sample number: "+str(total_samplenum))
print("total test number: "+str(total_testnum))
print("total noise number: "+str(noise_num))
file.close()