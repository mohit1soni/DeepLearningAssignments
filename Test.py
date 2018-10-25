import numpy as np
import matplotlib.pyplot as plt
N=50
x_old=list()
n=np.random.randint(1,N)
lamda1=np.sum([i for i in range(n)])/n
lamda2=(np.sum([i for i in range(N)])-n*lamda1)/(N-n)
print("lamda1: "+str(lamda1)+ " lambda2 : " + str(lamda2)+ " initial_n:" + str(n))
for i in range(N):
    if (i<= n and i>= 1):
        x=np.random.poisson(lamda1)
    else:
        x=np.random.poisson(lamda2)
    x_old.append(x)

x_new=list()
lam1=list()
lam2=list()
count=list()
for k in range(5200):
    for j in range(N):
        n=np.random.randint(1,N)
        lamda1=np.sum([i for i in range(n)])/n
        lamda2=(np.sum([i for i in range(N)])-n*lamda1)/(N-n)
        if (j<= n and j>= 1):
            x=np.random.poisson(lamda1)
        else:
            x=np.random.poisson(lamda2)
    if(k>200):
        x_new.append(x)
        lam1.append(lamda1)
        lam2.append(lamda2)
        count.append(k-200)
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.plot(x_old)
ax2=fig.add_subplot(2,1,1)
ax2.plot(count,lam1)
ax2.plot(count,lam2)
plt.show()