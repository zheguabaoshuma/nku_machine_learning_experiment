import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(25565)
center1=np.array([1,1])
center2=np.array([5,5])
sample1=np.random.randn(100,2)
sample2=np.random.randn(100,2)
sample1=sample1+center1
sample2=sample2+center2
tag1=np.ones(100)
tag2=np.ones(100)*(-1)

plt.scatter(sample1[:,0],sample1[:,1],marker='3',s=20)
plt.scatter(sample2[:,0],sample2[:,1],marker='3',s=20)
#plt.show()

def print_line(omega:np.ndarray,b:float,line_style:str='solid'):
    x=np.linspace(-2,8)
    y=(-b-omega[0]*x)/omega[1]
    plt.plot(x,y,linestyle=line_style)


class svm_classifier:
    def __init__(self, C=100.0, kernel='linear', max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.b=0
        self.epsilon=1e-5

        self._lambda=0
        self.beta=0.012
        self.eta=0.03

    def kernel_function(self, x1:np.ndarray, x2:np.ndarray):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'gaussian':
            sigma = 1.0
            return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

    def obj_function(self,x:np.ndarray,y:np.ndarray,alpha:np.ndarray):
        res:float=0

        return res

    def alpha2omega(self,alpha:np.ndarray,x:np.ndarray,y:np.ndarray)->np.ndarray:
        omega=np.zeros(len(x[0]))
        for i in range(0,len(x)):
            omega+=alpha[i]*y[i]*x[i]
        return omega

    def refresh_E(self,x:np.ndarray,y:np.ndarray):
        Gx:np.ndarray=(self.alpha*y)*np.sum(x@x.T,axis=0)+self.b
        self.E:np.ndarray=Gx-y

    def selectJ(self,i:int,x:np.ndarray,y:np.ndarray,length:int):
        j:int=np.random.randint(0,length)
        while i==j:
            j=np.random.randint(0,length)

        random_factor=np.random.randint(0,100)
        maxK = -1
        maxDeltaE = 0
        idx=np.nonzero(self.alpha)[0]
        if(len(idx)!=0):
            for index in idx:
                if index==i:continue
                Eindex=self.g(x,index,y)-y[index]
                Ei=self.g(x,i,y)-y[i]
                # eta = self.kernel_function(x[i], x[i]) + self.kernel_function(x[index], x[index]) - 2 * self.kernel_function(x[i], x[index])
                eta=1
                deltaE=abs(Ei-Eindex)/eta
                if (deltaE > maxDeltaE):
                    maxK = index
                    maxDeltaE = deltaE
            # print('choose '+str(maxK))
            print(random_factor)
            if random_factor%7==0:
                print('random choice')
                return j
            else:
                return maxK
        return j

    def selectJ_(self,i:int,x:np.ndarray,y:np.ndarray,length:int):
        j:int=np.random.randint(0,length)
        while i==j:
            j=np.random.randint(0,length)

        self.refresh_E(x,y)
        self.E[i] = self.g(x, i, y) - y[i]
        E_minus:np.ndarray=abs(self.E- self.E[i])
        E_minus_valid:np.ndarray=E_minus[np.nonzero(E_minus)]
        if len(E_minus_valid)!=0:
            j=E_minus.argmax()
        return j

    def update_b(self,i:int,j:int,x:np.ndarray,y:np.ndarray):
        res1:float=y[i]
        res2:float=y[j]
        #if  and
        for k in range(0,len(x)):
            res1-=self.alpha[k]*y[k]*self.kernel_function(x[k],x[i])
            res2-=self.alpha[k]*y[k]*self.kernel_function(x[k],x[j])

        if -self.epsilon < self.alpha[i] < self.C+self.epsilon:
            self.b=res1
        elif -self.epsilon < self.alpha[j] < self.C+self.epsilon:
            self.b=res2
        else:
            self.b=(res1+res2)/2

    def g(self,x:np.ndarray,i:int,y:np.ndarray):
        res:float=0
        index=[index for index,value in enumerate(self.alpha) if value != 0]
        for k in index:
            res+=self.alpha[k]*y[k]*self.kernel_function(x[i],x[k])
        return res+self.b

    def update(self,x:np.ndarray,y:np.ndarray,i:int,j:int=-1):
        alpha_update:bool=False

        if j==-1:
            j = self.selectJ(i, x,y,len(x))
        eta = self.kernel_function(x[i], x[i]) + self.kernel_function(x[j], x[j]) - 2 * self.kernel_function(x[i], x[j])
        if eta == 0: return -2

        self.E[i] = self.g(x, i, y) - y[i]
        self.E[j] = self.g(x, j, y) - y[j]

        if y[i]!=y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[j] + self.alpha[i] - self.C)
            H = min(self.C, self.alpha[j] + self.alpha[i])
        alpha_newj = \
            self.alpha[j] + (1) * y[j] * (self.E[i] - self.E[j]) / eta
        alpha_oldj = self.alpha[j]

        if alpha_newj > H:
            self.alpha[j] = H
        elif alpha_newj < L:
            self.alpha[j] = L
        else:
            self.alpha[j] = alpha_newj

        self.alpha[i] = self.alpha[i] + y[i] * y[j] * (alpha_oldj - self.alpha[j])
        self.update_b(i, j, x, y)

        self.E[i]=self.g(x,i,y)-y[i]
        self.E[j]=self.g(x, j, y) - y[j]
        if np.sum(np.abs(alpha_newj - self.alpha[j])) < 1e-10:
            return -2#-2 means tiny update, time to change a new strategy
        return 1#normal status

    def check_kkt(self,x:np.ndarray,i:int,y:np.ndarray)->bool:
        kkt=False
        check_kkt=y[i]*self.g(x,i,y)
        if (check_kkt>1-self.epsilon and -self.epsilon < self.alpha[i] < self.epsilon )or \
                (1-self.epsilon < check_kkt < 1+self.epsilon and -self.epsilon < self.alpha[
            i] < self.C + self.epsilon) or \
                (check_kkt<1+self.epsilon and self.C - self.epsilon < self.alpha[i] < self.C + self.epsilon):
            kkt=True
        return kkt

    def all_check_kkt(self,x:np.ndarray,y:np.ndarray)->bool:
        kkt=False
        Gx:np.ndarray=self.alpha*y@(x@x.T)+self.b
        check_kkt:np.ndarray=y*Gx-1
        kkt_unsatisfy1=[idx for idx,value in enumerate(self.alpha) if check_kkt[idx]>-self.epsilon and (-self.epsilon>value or value>self.epsilon)]
        kkt_unsatisfy2=[idx for idx,value in enumerate(self.alpha) if self.epsilon>check_kkt[idx]>-self.epsilon and (value<self.epsilon or value>self.C-self.epsilon)]
        kkt_unsatisfy3=[idx for idx,value in enumerate(self.alpha) if check_kkt[idx]<self.epsilon and (self.C-self.epsilon<value or value<self.C+self.epsilon)]
        if len(kkt_unsatisfy1)==0 and len(kkt_unsatisfy2)==0 and len(kkt_unsatisfy3)==0:
            return True
        else: return False

    def train(self,x:np.ndarray,y:np.ndarray,max_internum:int):
        self.alpha=np.zeros(len(x))
        self.omega=self.alpha2omega(self.alpha,x,y)
        self.E=np.zeros(len(x))
        iternum:int=0
        alpha_update: bool = False
        while(iternum<max_internum):
            if (not alpha_update) or iternum==1:
                alpha_update=True
                for i1 in range(0,len(x)):
                    if not self.check_kkt(x,i1,y):
                        self.update(x,y,i1)
            if alpha_update:
                alpha_update=False
                for i2 in range(0,len(x)):
                    if (self.epsilon>self.alpha[i2]>-self.epsilon and y[i2]*self.g(x,i2,y)<1+self.epsilon):
                        status:int=self.update(x,y,i2)
                        if status==1:
                            alpha_update=True
                    elif (self.C+self.epsilon>self.alpha[i2]>self.C-self.epsilon and y[i2]*self.g(x,i2,y)>1-self.epsilon):
                        status:int=self.update(x,y,i2)
                        if status==1:
                            alpha_update=True
            iternum+=1
            print(iternum)
            self.omega: np.ndarray = self.alpha2omega(self.alpha, x, y)

        self.mark_sv(x,y)
            #print_line(self.omega,self.b)

    def _train_lagrange(self,x:np.ndarray,y:np.ndarray,iternum:int):
        iter=0
        yi=self.eta
        x1=np.array([value*y[idx] for idx,value in enumerate(x)])
        self.alpha=np.zeros(len(x))
        while iter<iternum:
            alpha_old=self.alpha
            alpha_new=self.alpha-yi/len(x)*(x1@x1.T@self.alpha-1+self._lambda*y+self.beta*y@self.alpha*y)
            alpha_new=np.where(alpha_new>self.C,self.C,alpha_new)
            alpha_new=np.where(alpha_new<0,0,alpha_new)
            self.alpha=alpha_new
            _lambda_old=self._lambda
            self._lambda=_lambda_old+self.beta/len(x)*(y@self.alpha)
            iter+=1
            if np.sum(abs(alpha_old-self.alpha))<1e-6:
                break
            print(np.sum(abs(alpha_old-self.alpha)))

        self.omega=self.alpha2omega(self.alpha,x,y)
        print(self.alpha)
        a1=np.array([value for idx,value in enumerate(self.alpha) if self.C>value>0])
        a2=np.array([value for idx,value in enumerate(a1) if np.max(a1)-0.00001>value>np.min(a1)+0.00001])
        if len(a2)==0:
            a2=np.array([a1[0]])
        op=np.median(a2)
        # op1=np.min(abs(a2-op))
        j=np.argmin(abs(a2-op))
        op=a2[j]
        j=np.argmin(abs(self.alpha-op))
        u=0
        for _ in range(len(x)):
            u+=self.alpha[_]*y[_]*(x[_]@x[j])
        self.b=y[j]-u
        self.mark_sv(x,y)
        print_line(self.omega,self.b)


    def mark_sv(self,x:np.ndarray,y:np.ndarray):
        # for i in range(0,len(x)):
        #     if(self.epsilon>y[i]*(self.omega@x[i]+self.b)-1>-self.epsilon):
        #         plt.scatter(x[i][0],x[i][1],s=50,c='none',marker='o',edgecolors='red')
        for idx,alpha in enumerate(self.alpha):
            # if alpha==self.C:
            #     plt.scatter(x[idx][0], x[idx][1], s=50, c='none', marker='o', edgecolors='blue')
            if 0<alpha<self.C:
                plt.scatter(x[idx][0], x[idx][1], s=50, c='none', marker='o', edgecolors='blue')
            elif alpha==self.C:
                plt.scatter(x[idx][0], x[idx][1], s=50, c='none', marker='o', edgecolors='red')
        print_line(self.omega,self.b-1,'dashdot')
        print_line(self.omega,self.b+1,'dashdot')

svm=svm_classifier()
# svm.train(np.concatenate([sample1,sample2]),np.concatenate([tag1,tag2]),10000)
start=time.time()
svm._train_lagrange(np.concatenate([sample1,sample2]),np.concatenate([tag1,tag2]),1000000)
end=time.time()
print(svm.omega)
print(svm.alpha)
print(svm.b)
# print_line(svm.omega,svm.b)
print("model eta: "+str(svm.eta/(2*len(sample1))))
print("model beta: "+str(svm.beta))
print("time consume: "+str(end-start)+'s')
plt.show()