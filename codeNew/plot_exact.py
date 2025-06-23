import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sys 

import pandas as pd

rfs = ['SJ','MJ']

if len(sys.argv) > 2 and sys.argv[1] in rfs: 
    rf = sys.argv[1]
    contrast = float(sys.argv[2])
    print("Generating data for the reference solution = {} and contrast {}".format(rf,contrast))
else: 
    raise ValueError('Please provide one of the following options for the reference solution {} and a contrast'.format(rfs))


def conditional(x,eta,true,false):
    if x <= eta: 
        return true
    else: 
        return false

def singleJump_refsol(xu,t,c1,c2,w1,pos=0.5,dt=False):
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

def multipleJump_refsol(t,xu,c1,c2,w1,pos,n=3,dt=False):
    w2 = c1*w1/c2
    pos2 = (2*np.pi*n + w2*pos)/w2
    w3 = w1
    c3 = c1
    if not dt:
        u1 = np.cos(w1*c1*t) *np.cos(w1*(xu-pos))
        u2 = np.cos(w2*c2*t)*np.cos(w2*(xu-pos))
        u3 = np.cos(w3*c3*t)*np.cos(w3*(xu-pos2))
        sol = conditional(xu,pos2,conditional(xu,pos,u1,u2),u3)
        #sol = ufl.conditional(ufl.le(xu-pos2,0.0),ufl.conditional(ufl.le(xu-pos,0.0),u1,u2),u3)
    else: 
        u1 = -w1*c1*np.sin(w1*c1*t) *np.cos(w1*(xu-pos))
        u2 = -w2*c2*np.sin(w2*c2*t)*np.cos(w2*(xu-pos))
        u3 = -w3*c3*np.sin(w3*c3*t)*np.cos(w3*(xu-pos2))
        sol = conditional(xu,pos2,conditional(xu,pos,u1,u2),u3)
        #sol = ufl.conditional(ufl.le(xu-pos2,0.0),ufl.conditional(ufl.le(xu-pos,0.0),u1,u2),u3)
    return sol
    

def get_params(rf,contrast):
    if rf == 'SJ':
        w1 = 3*np.pi
        c1 = contrast
        c2 = 1
        pos= 0.5 
    elif rf == 'MJ': 
        c1 = contrast
        c2 = 1

        #weights
        w1 = 3*np.pi
        w2 = c1*w1/c2

        pos= 0.4
    return w1,c1,c2,pos




def sol(t,xu,contrast):
    w1,c1,c2,pos = get_params(rf,contrast)
    return singleJump_refsol(xu=xu,t=t,c1=c1,c2=c2,w1=w1,pos=pos,dt=False) if rf == 'SJ' else multipleJump_refsol(t,xu,c1,c2,w1,pos,n=3 if contrast == 7.5 else 4,dt=False)

def dt_sol(t,xu,contrast):
    w1,c1,c2,pos = get_params(rf,contrast)
    return singleJump_refsol(xu=xu,t=t,c1=c1,c2=c2,w1=w1,pos=pos,dt=True) if rf == 'SJ' else multipleJump_refsol(t,xu,c1,c2,w1,pos,n=3 if contrast == 7.5 else 4,dt=True)



def plot(t,save_data=False):
    w1,c1,c2,pos = get_params(rf,contrast)
    x_pts = np.linspace(0.0,1.0,num=500).tolist()
    y_pts = np.zeros(len(x_pts))

    for i in range(len(x_pts)):
        y_pts[i] = sol(t,x_pts[i],contrast)

    if save_data: 
        data = {'x':x_pts,'y':y_pts}
        df = pd.DataFrame(data)
        df.to_csv('../dataNew/plots/exact_plot{}_contrast{}_time{}.csv'.format(rf,c1,t),index=False)
        

    plt.plot(x_pts,y_pts)
    plt.show()

plot(0.25,True)