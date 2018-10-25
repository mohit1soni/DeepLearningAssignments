import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
# generating the clipped normal distribution
mean=5
standard_deviation=3
Range=[-100,100]
max_iteration=5000

# Data Generation for the comparision
x_sync=truncnorm.rvs(Range[0],Range[1],5,3,size=10000)

# Defining the Metropolis Algorithm for random walk having symmetry
def random_walk(x_initial,sd_q,max_iteration,Range):
    """ This model takes the input the initial guess of the parameter, wanted standard deviation for
    markov sampling, maximum iteration to run and the range in which normal distribution is defined"""
    x_new=list()
    x_new.append(x_initial)
    x_state=x_initial #This state variable will be updated if the generated probabiliy is favourable.
    for i in range(max_iteration):
        x_star=np.random.normal(x_state,1) # Probability distribution for Markov Sampling
        limit=np.random.uniform(low=0,high=1) #Generating Data form uniform distribution

        if(i<100):
            mean_ap=x_state
        else:
            mean_ap =np.mean(x_new)
        if limit < min(1,truncnorm.pdf(x_star,Range[0],Range[1],loc=mean_ap,scale=3)/truncnorm.pdf(x_state,Range[0],Range[1],loc=mean_ap,scale=3)):
            x_state=x_star
        else:
            x_state=x_state
        x_new.append(x_state)

    return x_new


#  Testing the Hastings Algorithm of Random Walk.............
initial_mean=2
sd_q=1
x_new=random_walk(initial_mean,sd_q,max_iteration,Range)


fig=plt.figure()
fig.add_subplot(121)
plt.hist(x_new[int(max_iteration*0.1):max_iteration])
plt.title("Estimated_Distribution_n_iteration:="+str(max_iteration))
fig.add_subplot(122)
plt.hist(x_sync)
plt.title("Real Distribution")
plt.savefig("Figures/Metropolis_Rand/Distibution_plot_n_iter_"+str(max_iteration)+".png")
plt.show()

