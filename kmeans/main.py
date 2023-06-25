import numpy as np
import matplotlib.pyplot as plt

default_color= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
np.random.seed(25565)
samples=np.random.rand(2000,2)*9
samples=samples.tolist()
tags=[]
center_index=np.random.randint(0,2000,9)
centers=[samples[i] for i in center_index]
for C in centers:
    plt.scatter(C[0],C[1],s=30,marker='s')
centers=np.array(centers)

for idx,point in enumerate(samples):
    if 3>point[0]>0 and 3>point[1]>0:
        tags.append(0)
        plt.scatter(point[0],point[1],marker='3',c=default_color[0],s=20)
    elif 6>point[0]>3 and 3>point[1]>0:
        tags.append(1)
        plt.scatter(point[0],point[1],marker='3',c=default_color[1],s=20)
    elif 9>point[0]>6 and 3>point[1]>0:
        tags.append(2)
        plt.scatter(point[0],point[1],marker='3',c=default_color[2],s=20)
    elif 3>point[0]>0 and 6>point[1]>3:
        tags.append(3)
        plt.scatter(point[0],point[1],marker='3',c=default_color[3],s=20)
    elif 6>point[0]>3 and 6>point[1]>3:
        tags.append(4)
        plt.scatter(point[0],point[1],marker='3',c=default_color[4],s=20)
    elif 9>point[0]>6 and 6>point[1]>3:
        tags.append(5)
        plt.scatter(point[0],point[1],marker='3',c=default_color[5],s=20)
    elif 3>point[0]>0 and 9>point[1]>6:
        tags.append(6)
        plt.scatter(point[0],point[1],marker='3',c=default_color[6],s=20)
    elif 6>point[0]>3 and 9>point[1]>6:
        tags.append(7)
        plt.scatter(point[0],point[1],marker='3',c=default_color[7],s=20)
    elif 9>point[0]>6 and 9>point[1]>6:
        tags.append(8)
        plt.scatter(point[0],point[1],marker='3',c=default_color[8],s=20)

samples=np.array(samples)
tags=np.array(tags)

def kmeans_train(iternum:int,x:np.ndarray,y:np.ndarray,centers:np.ndarray):
    iter=0
    pred_tags=np.zeros(len(x))
    while iter<iternum:
        for idx,point in enumerate(x):
            delta:np.ndarray=centers-point
            distance:np.ndarray=np.sum(np.abs(delta),axis=1)
            pred_tags[idx]=distance.argmin()
        for idx,center in enumerate(centers):
            all_points=[value for index,value in enumerate(x) if pred_tags[index]==idx]
            if len(all_points)!=0:
                avg_center=sum(all_points)/len(all_points)
                centers[idx]=avg_center
            else: continue
        iter+=1
        print(iter)
    for idx,c in enumerate(centers):
        plt.scatter(c[0],c[1],marker='o',s=30,c=default_color[idx])
kmeans_train(5000,samples,tags,centers)
plt.show()
