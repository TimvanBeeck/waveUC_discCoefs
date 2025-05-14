import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def conditional(x,eta,true,false):
    if x <= eta: 
        return true
    else: 
        return false

def new_refsol(xu,t,c1,c2,w1,pos=0.5,dt=False):
        w2 = c1*w1/c2
        if not dt:
            u1 = np.cos(w1*c1*t) *np.cos(w1*(xu-pos))
            u2 = np.cos(w2*c2*t)*np.cos(w2*(xu-pos))
            sol = conditional(xu,pos,u1,u2)
        else: 
            u1 = -w1*c1*np.sin(w1*c1*t) *np.cos(w1*(xu-pos))
            u2 = -w2*c2*np.sin(w2*c2*t)*np.cos(w2*(xu-pos))
            sol = conditional(xu,pos,u1,u2)
        return sol
    
w1 = 3*np.pi
c1 = 2.5
c2 = 1
pos= 0.5 


def sol(t,xu):
    return new_refsol(xu=xu,t=t,c1=c1,c2=c2,w1=w1,pos=pos,dt=False)

def dt_sol(t,xu):
    return new_refsol(xu=xu,t=t,c1=c1,c2=c2,w1=w1,pos=pos,dt=True)


def plot(t):
    x_pts = np.linspace(0.0,1.0,num=5000).tolist()


    y_pts = np.zeros(len(x_pts))

    for i in range(len(x_pts)):
        y_pts[i] = sol(t,x_pts[i])

    plt.plot(x_pts,y_pts)
    plt.show()

plot(0.5)