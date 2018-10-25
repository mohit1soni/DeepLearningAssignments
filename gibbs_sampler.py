import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.animation as animation
#  This program is to explain the gibbs sampling using bi_variate Normal Distribution.
rho=0.9
cov_matrix=[1,rho,rho,1]
cov_matrix=np.reshape(cov_matrix,(2,2))
mean=[0,0]
synt_dist=multivariate_normal(mean,cov_matrix)

"""This function describes the basic procedure of gibbs sampling
using conditional prob of distribution of variables"""

def gibbs_sampler(rho,theta_initial,max_iteration=10):
    """ Input provided to this function is the rho value for variance calculation of individual conditional
    probabilities
        theta_initial is the initial_guess of the estimate where you want your gibbs sampler to start
        max_iteration is the maximum number of iteration you want to run your gibbs sampler."""
    var_1=list()
    var_2=list()
    var_1.append(theta_initial[0])
    var_2.append(theta_initial[1])
    # These variable will maintain the values calculated using gibbs sampling for n time_steps
    daviation1=(1-rho**2)**0.5
    daviation2=(1-rho**2)**0.5
    for i in range(max_iteration):
        theta1=np.random.normal(rho*var_2[i],daviation1)
        theta2=np.random.normal(rho*var_1[i],daviation2)
        var_1.append(theta1)
        var_2.append(theta2)
    ans=[0,0]
    ans[0]=np.sum(var_1[int(max_iteration/2):max_iteration])/(max_iteration/2)
    ans[1]=np.sum(var_2[int(max_iteration/2):max_iteration])/(max_iteration/2)
    return ans,var_1,var_2

theta_initial=[-3,3]
iteration=10000
estimate,pred_x,pred_y=gibbs_sampler(rho,theta_initial,max_iteration=iteration)
print(estimate)

x,y=np.mgrid[-3:3:0.1,-3:3:0.1]
positions=np.dstack((x,y))
fig=plt.figure()
plt.contourf(x,y,synt_dist.pdf(positions),colors=('w','k'))
plt.title("gibbs_sampling_"+str(iteration)+"_"+str(estimate[0])+"_"+str(estimate[1]))
plt.plot(pred_x,pred_y,'ro')
# for i in range(iteration):
#     plt.plot(pred_x[i],pred_y[i],'ro')
#     plt.pause(0.000001)
plt.axis([-3.1,3.1,-3.1,3.1])
plt.savefig("Figures/gibbs_sampling/gibbs_normal_"+str(iteration)+".png")
plt.show()