import numpy as np
import matplotlib.pyplot as plt

def draw_line(omega:np.ndarray,bias:np.ndarray,color:str=""):
    x=np.linspace(0,5)
    y=(-bias-omega[0]*x)/omega[1]
    if color=="":
        plt.plot(x,y)
    else:
        plt.plot(x,y,color=color)

def penalty_function(omega_and_bias:np.ndarray,data_x:np.ndarray,data_y:np.ndarray):
    return np.sum(1/data_y.size*(omega_and_bias[0:-1]@data_x.T+omega_and_bias[-1]-data_y)**2)#+0.5*(omega_and_bias@omega_and_bias.T)

def partial(func,omega_and_bias:np.ndarray,data_x:np.ndarray,data_y:np.ndarray,partial_seq:int):
    delta=np.zeros(omega_and_bias.shape)
    delta[partial_seq]+=0.001
    f1=func(omega_and_bias,data_x,data_y)
    #print(f1)
    f2=func(omega_and_bias+delta,data_x,data_y)
    return (f2-f1)/0.001

def gradient(func,omega_and_bias:np.ndarray,data_x:np.ndarray,data_y:np.ndarray)->np.ndarray:
    grad=np.zeros(omega_and_bias.shape)
    for c in range(0,len(grad)):
        grad[c]=partial(func,omega_and_bias,data_x,data_y,c)
    return grad

def gradient_descent(func,omega_and_bias:np.ndarray,data_x:np.ndarray,data_y:np.ndarray):
    step=1e-5
    iternum=0
    while(iternum<=10000):
        nablaf = gradient(func, omega_and_bias, data_x, data_y)
        update:np.ndarray=nablaf*step
        if iternum%200==0 and False:
            draw_line(omega_and_bias[0:2],omega_and_bias[2])
        if update.max()<1e-7:
            print(iternum)
            break
        omega_and_bias-=update
        iternum+=1

def objective_function(omega_and_bias:np.ndarray,data_x:np.ndarray,data_y:np.ndarray):#data is a single sample
    return data_y*(omega_and_bias[0:-1]@data_x.T+omega_and_bias[-1])
def optmize(omega_and_bias:np.ndarray,data_x:np.ndarray,data_y:np.ndarray):
    step=1e-5
    iternum=0
    while iternum<10000:
        t:np.ndarray=[]
        for i in range(0,len(data_x)):
            if objective_function(omega_and_bias,data_x[i],data_y[i])<=0 or True:
                update=np.concatenate((step*data_y[i]*data_x[i],np.asarray([step*data_y[i]])),axis=0)
                omega_and_bias+=update
        t=objective_function(omega_and_bias,data_x,data_y)
        if t.min()>0:
            print(iternum)
            break
        iternum+=1
    print(iternum)

def validate(omega_and_bias:np.ndarray,point:np.ndarray):
    if omega_and_bias[0:-1]@point.T+omega_and_bias[-1]>=0:
        return 1
    else: return-1

#center1=np.array([1,1])
center1=np.ones(100)*-10
#center2=np.array([4,4])
center2=np.ones(100)*10
np.random.seed(25565)

x1:list=[]
x2:list=[]
x_test:list=[]
#generate training data
for i in range(0,100):
    ran=np.random.normal(0,0.1,[100,])
    ran=ran+center1
    x1.append(ran)

for i in range(0,100):
    ran=np.random.normal(0,0.1,[100,])
    ran=ran+center2
    x2.append(ran)
#generate test data
for i in range(0,10):
    ran=np.random.normal(0,0.1,[100,])
    ran=ran+center1
    x_test.append(ran)

for i in range(0,10):
    ran=np.random.normal(0,0.1,[100,])
    ran=ran+center2
    x_test.append(ran)


x1=np.asarray(x1)
x2=np.asarray(x2)
x_test=np.asarray(x_test)
y_test=np.ones(len(x_test))
y_test[10:]*=(-1)

sample_x1=x1[:,0]
samply_y1=x1[:,1]
plt.scatter(sample_x1,samply_y1,marker="1")

sample_x2=x2[:,0]
samply_y2=x2[:,1]
plt.scatter(sample_x2,samply_y2,marker="2")

test_x1=x_test[:10,0]
test_y1=x_test[:10,1]
plt.scatter(test_x1,test_y1,marker="3")

test_x2=x_test[10:20,0]
test_y2=x_test[10:20,1]
plt.scatter(test_x2,test_y2,marker="3")

x1_sign=np.ones(len(x1))
x2_sign=np.ones(len(x2))*(-1)

omega=np.random.random(100)#100d data
bias=np.random.random(1)
#draw_line(omega,bias,color="green")

omega_bias=np.concatenate((omega,bias),axis=0)
gradient_descent(penalty_function,omega_bias,np.concatenate((x1,x2)),np.concatenate((x1_sign,x2_sign)))
#optmize(omega_bias,np.concatenate((x1,x2)),np.concatenate((x1_sign,x2_sign)))
#draw_line(omega_bias[0:2],omega_bias[2],"brown")

correct=0
for i in range(0,len(x_test)):
    pred=validate(omega_bias,x_test[i])
    if pred==y_test[i]:correct+=1
print("acc is "+str(correct/len(x_test)))

#plt.show()