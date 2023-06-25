import numpy as np
import matplotlib.pyplot as plt

sample:list=[]
test:list=[]

def sort_key(item:tuple):
    return item[0]

def knn_predict(point:np.array,n:int)->int:
    dist:list=[]
    for k in range(1,len(sample)+1):
        kind_sample=sample[k-1]
        for i in range(0,len(kind_sample)):
            eu_dis=(kind_sample[i]-point)@(kind_sample[i]-point)
            dist.append((eu_dis,k))
    dist.sort(key=sort_key)
    vote:list=dist[0:n]
    result=np.zeros(10)
    for j in range(0,len(vote)):
        result[vote[j][1]]+=1
    return result.argmax()

x_kind1=np.random.rand(100,2)*3#kind1 x in [0,3], y in [0,3]
x_kind1_test=np.random.rand(500,2)*3
sample.append(x_kind1)
test.append(x_kind1_test)
plt.scatter(x_kind1[:,0],x_kind1[:,1],marker='3')
#plt.scatter(x_kind1_test[:,0],x_kind1_test[:,1],marker='.',c='#1f1e33',alpha=0.5)

x_kind2=np.random.rand(100,2)*3+np.array([3.5,0])#kind2 x in [3.5,6.5], y in [0,3]
x_kind2_test=np.random.rand(500,2)*3+np.array([3.5,0])
sample.append(x_kind2)
test.append(x_kind2_test)
plt.scatter(x_kind2[:,0],x_kind2[:,1],marker='3')
#plt.scatter(x_kind2_test[:,0],x_kind2_test[:,1],marker='.',c='#1f1e33',alpha=0.5)

x_kind3=np.random.rand(100,2)*3+np.array([7,0])#kind3 x in [7,10], y in [0,3]
x_kind3_test=np.random.rand(500,2)*3+np.array([7,0])
sample.append(x_kind3)
test.append(x_kind3_test)
plt.scatter(x_kind3[:,0],x_kind3[:,1],marker='3')
#plt.scatter(x_kind3_test[:,0],x_kind3_test[:,1],marker='.',c='#1f1e33',alpha=0.5)

x_kind4=np.random.rand(100,2)*3+np.array([0,3.5])#kind4 x in [0,3], y in [3.5,6.5]
x_kind4_test=np.random.rand(500,2)*3+np.array([0,3.5])
sample.append(x_kind4)
test.append(x_kind4_test)
plt.scatter(x_kind4[:,0],x_kind4[:,1],marker='3')
#plt.scatter(x_kind4_test[:,0],x_kind4_test[:,1],marker='.',c='#1f1e33',alpha=0.5)

x_kind5=np.random.rand(100,2)*3+np.array([3.5,3.5])#kind5 x in [3.5,6.5], y in [3.5,6.5]
x_kind5_test=np.random.rand(500,2)*3+np.array([3.5,3.5])
sample.append(x_kind5)
test.append(x_kind5_test)
plt.scatter(x_kind5[:,0],x_kind5[:,1],marker='3')
#plt.scatter(x_kind5_test[:,0],x_kind5_test[:,1],marker='.',c='#1f1e33',alpha=0.5)

x_kind6=np.random.rand(100,2)*3+np.array([7,3.5])#kind6 x in [7,10], y in [3.5,6.5]
x_kind6_test=np.random.rand(500,2)*3+np.array([7,3.5])
sample.append(x_kind6)
test.append(x_kind6_test)
plt.scatter(x_kind6[:,0],x_kind6[:,1],marker='3')
#plt.scatter(x_kind6_test[:,0],x_kind6_test[:,1],marker='.',c='#1f1e33',alpha=0.5)

x_kind7=np.random.rand(100,2)*3+np.array([0,7])#kind7 x in [0,3], y in [7,10]
x_kind7_test=np.random.rand(500,2)*3+np.array([0,7])
sample.append(x_kind7)
test.append(x_kind7_test)
plt.scatter(x_kind7[:,0],x_kind7[:,1],marker='3')
#plt.scatter(x_kind7_test[:,0],x_kind7_test[:,1],marker='.',c='#1f1e33',alpha=0.5)

x_kind8=np.random.rand(100,2)*3+np.array([3.5,7])#kind8 x in [3.5,6.5], y in [7,10]
x_kind8_test=np.random.rand(500,2)*3+np.array([3.5,7])
sample.append(x_kind8)
test.append(x_kind8_test)
plt.scatter(x_kind8[:,0],x_kind8[:,1],marker='3')
#plt.scatter(x_kind8_test[:,0],x_kind8_test[:,1],marker='.',c='#1f1e33',alpha=0.5)

x_kind9=np.random.rand(100,2)*3+np.array([7,7])#kind9 x in [7,10], y in [7,10]
x_kind9_test=np.random.rand(500,2)*3+np.array([7,7])
sample.append(x_kind9)
test.append(x_kind9_test)
plt.scatter(x_kind9[:,0],x_kind9[:,1],marker='3')
#plt.scatter(x_kind9_test[:,0],x_kind9_test[:,1],marker='.',c='#1f1e33',alpha=0.5)

correct=0
default_color= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i in range(1,10):
    current_test=test[i-1]
    for k in range(0,len(current_test)):
        ans=knn_predict(current_test[k],10)
        plt.scatter(current_test[k][0],current_test[k][1], marker='s', c=default_color[i-1], alpha=0.5,s=15)
        if ans==i:
            correct+=1

plt.show()
print("acc is "+str(correct/(9*500)))
    
