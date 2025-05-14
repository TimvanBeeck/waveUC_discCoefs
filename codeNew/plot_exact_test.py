import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator, LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Testfile to check the implementation of the exact solution..., exchanged ufl.conditional with a function that should do the same, ufl.exp to np.exp and x[0] -> x as before...

def conditional(x,eta,true,false):
    if x <= eta: 
        return true
    else: 
        return false

def ref_sol(x,t,p0,c1,c2,trunc,pos,dt=False):
    if not dt:
        #import pdb; pdb.set_trace()
        u1_plus = sum([((c2-c1)/(c2+c1))**k * (p0(k+x-c1*t)-p0(k-x-c1*t)) for k in range(trunc)])
        u1_minus = ((1/pos*c1)/(c2+c1))*sum([((c2-c1)/(c2+c1))**k * p0(c1/c2*(x-pos)+k+pos-c1*t) for k in range(trunc)])
    else:
        u1_plus = sum([((c2-c1)/(c2+c1))**k * (800*c1*(k+x-c1*t-1/5)*p0(k+x-c1*t)-800*c1*(k-x-c1*t-1/5)*p0(k-x-c1*t)) for k in range(trunc)])
        u1_minus = sum([((c2-c1)/(c2+c1))**k * 800*c1*(c1/c2*(x-pos)+k+pos-c1*t-1/5)*p0(c1/c2*(x-pos)+k+pos-c1*t) for k in range(trunc)])
    sol = conditional(x,pos,u1_plus,u1_minus)
    return sol

p0 = lambda x : 1/100*np.exp(-(20*(x-1/5))**2)

c1 = 1
c2 = 0.1
pos= 0.5
t = 0.0
trunc = 100

def sol(t,xu):
    return ref_sol(xu,t,p0,c1,c2,trunc,pos,dt=False)

def dt_sol(t,xu):
    return ref_sol(xu,t,p0,c1,c2,trunc,pos,dt=True)


def plot(t):
    x_pts = np.linspace(0.0,1.0,num=5000).tolist()


    y_pts = np.zeros(len(x_pts))

    for i in range(len(x_pts)):
        y_pts[i] = sol(t,x_pts[i])

    plt.plot(x_pts,y_pts)
    plt.show()



#import pdb; pdb.set_trace()
#t = 0.0
#while t < 1.0: 
#    plot(t)
#    t += 0.1
plot(0.25)

generate_data = True
if generate_data: 
    def generate_plot_data(contrast,t=0.25,nx=500):
        x_pts = np.linspace(0.0,1.0,num=nx).tolist()
        y_pts = np.zeros(len(x_pts))
        y_pts_dt = np.zeros(len(x_pts))
        for i in range(len(x_pts)):
            y_pts[i] = ref_sol(x_pts[i],t=t,p0=p0,c1=contrast,c2=1,trunc=100,pos=0.5,dt=False)
            y_pts_dt[i] = ref_sol(x_pts[i],t=t,p0=p0,c1=contrast,c2=1,trunc=100,pos=0.5,dt=True)
        return x_pts, y_pts, y_pts_dt

    import pandas as pd

    data_df = pd.DataFrame()
    for contrast in [1.0,1.5,2.0,2.5,5.5]:
        x,y,y_dt = generate_plot_data(contrast)
        exact_data = {'x':x,'y':y,'y_dt':y_dt,'contrast':np.repeat(contrast,len(x))}
        data_df = pd.concat([data_df,pd.DataFrame.from_dict(exact_data)],ignore_index=True)
        #data_df.to_csv('../data/jumpCoefs/exact_plot_data.csv',index=False)
    
