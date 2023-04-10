import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# plotting stuff
# Set parameters for plots below--from MK
#rcParams['font.family'] = 'serif'
#rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['xtick.major.size']=8
rcParams['ytick.major.size']=8
rcParams['xtick.minor.size']=3.5
rcParams['ytick.minor.size']=3.5
rcParams['xtick.labelsize']=10
rcParams['ytick.labelsize']=10
rcParams['contour.negative_linestyle'] = 'solid'



#### define colors for plotting below
BLUE = '#377eb8'
ORANGE = '#ff7f00'

PURP = '#984ea3' 
GREEN = '#4dab4a'

"""
From:    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
"""
def moving_average(a, n=3):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

# collocation points, final PINN output + absolute error
x = np.linspace(-np.pi, np.pi, 513) 
y_final = np.loadtxt('checkpoint6.txt') 
err_final = np.loadtxt('checkpoint6_residuals.txt') 

# define true solution
ytrue = np.sin(x) + np.sin(5*x)/5**2 + np.sin(15*x)/15**2 + np.sin(55*x)/55**2

# load evolution during training of the 4 'active' Fourier modes in the problem
# then take rolling average 
a,b,c,d = np.loadtxt('FebResErr.txt', unpack=True)

roll_n = 50
a_avg = moving_average(a, roll_n)   # k = 1
b_avg = moving_average(b, roll_n)   # k = 5
c_avg = moving_average(c, roll_n)   # k = 15
d_avg = moving_average(d, roll_n)   # k = 55


plt.figure(1)
plt.plot(x,y_final)
plt.plot(x,ytrue)
#plt.plot(x,err_final)

plt.figure(2) 
plt.plot(x, np.abs(y_final-ytrue))
plt.title('pointwise absolute error') 

fig3 = plt.figure(3, figsize=(6.5,4.2))
#plt.plot(a)
#plt.plot(b)
#plt.plot(c)
#plt.plot(d)
plt.plot(a_avg, label='$k=1$', color='k', marker='o', markersize=10,    markevery=[25], fillstyle='none' )
plt.plot(b_avg, label='$k=5$', color=BLUE, marker='v', markersize=10,   markevery=[25], fillstyle='none' )
plt.plot(c_avg, label='$k=15$', color=GREEN, marker='^', markersize=10, markevery=[95], fillstyle='none' )
plt.plot(d_avg, label='$k=55$', color=PURP, marker='s', markersize=10,  markevery=[2105], fillstyle='none' )
plt.xscale('log')
plt.xlabel('Training iteration', fontsize=11)
plt.ylabel('$\widehat{e}_k$', fontsize=13)
plt.legend(loc='best', frameon=False)

plt.savefig('frequency_principle_poisson.pdf', bbox_inches='tight')


plt.show()
