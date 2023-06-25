import numpy as np
import matplotlib.pyplot as plt

np.random.seed(25565)
sample=np.random.rand(200,2)*10
tag=np.zeros(200)
test=np.random.rand(100,2)*10
test_tag=np.zeros(100)
default_color= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
noise=np.random.rand(2000,2)*10
noise_tag=np.random.randint(1,10,2000)


'''
center1=np.array([1.5,1])
sample1=np.random.randn(600,2)+center1
center2=np.array([5,1])
sample2=np.random.randn(600,2)+center2
center3=np.array([8.5,1])
sample3=np.random.randn(600,2)+center3
center4=np.array([1.5,5])
sample4=np.random.randn(600,2)+center4
center5=np.array([5,5])
sample5=np.random.randn(600,2)+center5
center6=np.array([8.5,5])
sample6=np.random.randn(600,2)+center6
center7=np.array([1.5,8.5])
sample7=np.random.randn(600,2)+center7
center8=np.array([5,8.5])
sample8=np.random.randn(600,2)+center8
center9=np.array([8.5,8.5])
sample9=np.random.randn(600,2)+center9
samplen=np.concatenate([sample1[0:100],sample2[0:100],sample3[0:100],sample4[0:100],sample5[0:100],sample6[0:100],sample7[0:100],sample8[0:100],sample9[0:100]])
testn=np.concatenate([sample1[100:600],sample2[100:600],sample3[100:600],sample4[100:600],sample5[100:600],sample6[100:600],sample7[100:600],sample8[100:600],sample9[100:600]])
ntag=[]
ntest_tag=[]
for k in range(0,9):
    plt.scatter(samplen[k*100:(k+1)*100,0],samplen[k*100:(k+1)*100,1],marker='3',color=default_color[k])
    plt.scatter(testn[k*500:(k+1)*500,0],testn[k*500:(k+1)*500,1],marker='s',alpha=0.5,s=15,color=default_color[k])
    ntag+=(100*[k+1])
    ntest_tag+=(500*[k+1])
plt.show()
'''
total_samplenum=0
total_noisenum=0
#plt.scatter(sample[:,0],sample[:,1])

def sort_key(item:tuple):
    return item[0]

def knn_predict(point:np.array,n:int)->int:
    dist:list=[]
    for k in range(1,len(sample)):
        if sample[k][0]==-1:
            continue
        eu_dis=(sample[k]-point)@(sample[k]-point)
        man_dis=np.sum(np.abs(sample[k]-point))
        inf_dis=np.max(np.abs(sample[k]-point))
        dist.append((inf_dis,int(tag[k])))
    dist.sort(key=sort_key)
    vote:list=dist[0:n]
    result=np.zeros(10)
    for j in range(0,len(vote)):
        result[vote[j][1]]+=1
    return result.argmax()

for i in range(0,len(sample)):
    if((sample[i][0]>=3 and sample[i][0]<=3.5) or (sample[i][0]>=6.5 and sample[i][0]<=7)
     or (sample[i][1]>=3 and sample[i][1]<=3.5) or (sample[i][1]>=6.5 and sample[i][1]<=7)):
        sample[i]=np.array([-1,-1])
        continue
    if(sample[i][0]>=0 and sample[i][0]<=3 and sample[i][1]>=0 and sample[i][1]<=3):
        tag[i]=1
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[0])
    elif(sample[i][0]>=3.5 and sample[i][0]<=6.5 and sample[i][1]>=0 and sample[i][1]<=3):
        tag[i]=2
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[1])
    elif(sample[i][0]>=7 and sample[i][0]<=10 and sample[i][1]>=0 and sample[i][1]<=3):
        tag[i]=3
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[2])
    elif(sample[i][0]>=0 and sample[i][0]<=3 and sample[i][1]>=3.5 and sample[i][1]<=6.5):
        tag[i]=4
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[3])
    elif(sample[i][0]>=3.5 and sample[i][0]<=6.5 and sample[i][1]>=3.5 and sample[i][1]<=6.5):
        tag[i]=5
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[4])
    elif(sample[i][0]>=7 and sample[i][0]<=10 and sample[i][1]>=3.5 and sample[i][1]<=6.5):
        tag[i]=6
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[5])
    elif(sample[i][0]>=0 and sample[i][0]<=3 and sample[i][1]>=7 and sample[i][1]<=10):
        tag[i]=7
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[6])
    elif(sample[i][0]>=3.5 and sample[i][0]<=6.5 and sample[i][1]>=7 and sample[i][1]<=10):
        tag[i]=8
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[7])
    elif(sample[i][0]>=7 and sample[i][0]<=10 and sample[i][1]>=7 and sample[i][1]<=10):
        tag[i]=9
        plt.scatter(sample[i][0],sample[i][1],marker='3',c=default_color[8])
    total_samplenum+=1

for i in range(0,len(noise)):
    if((noise[i][0]>=3 and noise[i][0]<=3.5) or (noise[i][0]>=6.5 and noise[i][0]<=7)
     or (noise[i][1]>=3 and noise[i][1]<=3.5) or (noise[i][1]>=6.5 and noise[i][1]<=7)):
        noise[i]=np.array([-1,-1])
        continue
    plt.scatter(noise[i][0],noise[i][1],marker='3',c=default_color[noise_tag[i]-1])
    total_noisenum+=1
sample=np.concatenate((sample,noise))
tag=np.concatenate((tag,noise_tag))

for i in range(0,len(test)):
    if((test[i][0]>=3 and test[i][0]<=3.5) or (test[i][0]>=6.5 and test[i][0]<=7)
     or (test[i][1]>=3 and test[i][1]<=3.5) or (test[i][1]>=6.5 and test[i][1]<=7)):
        test[i]=np.array([-1,-1])
        continue
    if(test[i][0]>=0 and test[i][0]<=3 and test[i][1]>=0 and test[i][1]<=3):
        test_tag[i]=1
        plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=3.5 and test[i][0]<=6.5 and test[i][1]>=0 and test[i][1]<=3):
        test_tag[i]=2
        plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=7 and test[i][0]<=10 and test[i][1]>=0 and test[i][1]<=3):
        test_tag[i]=3
        plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=0 and test[i][0]<=3 and test[i][1]>=3.5 and test[i][1]<=6.5):
        test_tag[i]=4
        plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=3.5 and test[i][0]<=6.5 and test[i][1]>=3.5 and test[i][1]<=6.5):
        test_tag[i]=5
        plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=7 and test[i][0]<=10 and test[i][1]>=3.5 and test[i][1]<=6.5):
        test_tag[i]=6
        plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=0 and test[i][0]<=3 and test[i][1]>=7 and test[i][1]<=10):
        test_tag[i]=7
        plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=3.5 and test[i][0]<=6.5 and test[i][1]>=7 and test[i][1]<=10):
        test_tag[i]=8
        plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
    elif(test[i][0]>=7 and test[i][0]<=10 and test[i][1]>=7 and test[i][1]<=10):
        test_tag[i]=9
        plt.scatter(test[i][0],test[i][1],marker='s',c='#1f1e33',s=15,alpha=0.5)
#plt.show()

correct=0
totalnum=0
for k in range(0,len(test)):
    if test[k][0]==-1:
         continue
    else:
        ans=knn_predict(test[k],25)
        plt.scatter(test[k][0],test[k][1],marker='s',c=default_color[int(ans-1)],s=15,alpha=0.5)
        totalnum+=1
        if ans==test_tag[k]:
            correct+=1
plt.show()

print('total sample number is '+str(total_samplenum))
print('total noise number is '+str(total_noisenum))
print('total test number is '+str(totalnum))
print('the acc is '+str(correct/totalnum))
