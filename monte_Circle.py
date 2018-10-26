import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
# This code is for the general understanding of the Monte-carlo pricess
num_smaples=10000# Give this according to your choice....
sq_x_max=1
sq_y_max=1

# Fuction for simulating the Monte Carlo mathod for the calculation of area_ration between square and rectangle
def Monte_carlo(num_smaples):
    rand_sample_x=list()
    rand_sample_y=list()
    count=0
    for i in range(num_smaples):
        rn=np.random.uniform(low=0,high=1)
        rm=np.random.uniform(low=0,high=1)
        rand_sample_x.append(rn)
        rand_sample_y.append(rm)
    for i in range(len(rand_sample_x)):
        if (rand_sample_x[i]-0.5)**2+(rand_sample_y[i]-0.5)**2<=0.25:
            count=count+1
    result=float(count/num_smaples)
    return result,rand_sample_x,rand_sample_y


area,x,y=Monte_carlo(num_smaples)
print("Area of the circle : " + str(area))
fig=plt.figure()
square=plt.Rectangle((0,0),sq_x_max,sq_y_max,fc='black')
circle=plt.Circle((0.5,0.5),radius=0.5,fc='y')
plt.gca().add_patch(square)
plt.gca().add_patch(circle)
plt.plot(x,y,'ro')
plt.axis([0,1,0,1])
plt.title("Area_from_montecarlo = "+str(area))
plt.savefig("Figures/Monte_Carlo_"+str(num_smaples)+".png")

