## IMPORTS ##
import numpy as np
from math import pi,sqrt  
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, dS,jump,div
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from dolfinx.mesh import create_unit_interval
from dolfinx.io import XDMFFile
import sys
sys.setrecursionlimit(10**6)
from GMREs import GMRes
from space_time import * 
from precomp_time_int import theta_ref, d_theta_ref 
import pypardiso
import time
import cProfile
import resource
import pandas as pd 

from Solve1D import *

import sys
args = ['fix','adap']

if len(sys.argv) > 1 and sys.argv[1] in args:
    TAdap = True if sys.argv[1] == 'adap' else False
    print("Solving with TAdap = ", TAdap)
else: 
    raise ValueError('Please give either fix or adap as arguments')


#Stabilization params
def set_stabs(c1,c2=1.0): 
    stabs = {#"data": 1e4,
                "data": 1e4,
                #"dual": 1.0,
                "dual": 1e0,
                #"primal": 1e-3,
                "primal": 1e-2,
                "primal-jump":1e-2, # for simple sol: 1e-2, 
                #"primal-jump-vel":1.0,
                "primal-jump-displ-grad":1e-2/max(c1**2,c2**2)**2 # for simple sol: 1e-2
            } 
    return stabs



df = pd.DataFrame()


## for multiple contrasts (well-posed=False, GCC = True)
orders = [3]
contrasts = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5]
ref_lvls = lambda order, T : [3]

#time
#TAdap = True
Tfunc = lambda contrast: 1/4*(1+1/contrast)*1.01 if TAdap else 0.5



make_plots=True

for order in orders:
    for contrast in contrasts: 
        T = Tfunc(contrast)
        print("Solving with time T = ", T)
        plot_df = pd.DataFrame()
        for ref_lvl in ref_lvls(order,T):
            print("Order = {}, ref_lvl = {}, contrast = {}".format(order,ref_lvl,contrast))
            stabs = set_stabs(c1=contrast)
            results = Solve(ref_lvl=ref_lvl,order=order,contrast=contrast,stabs=stabs,plot_sol=make_plots,plot_error=make_plots,T=T)
            if make_plots:
                errors = results[0]
                ba_errors = results[1]
                condest = results[2]
                plot_data_sol = results[3]
                plot_data_errors = results[4]
                iterations = results[5]
            else: 
                errors = results[0]
                ba_errors = results[1]
                condest = results[2]
                iterations = results[3]
            print("our condest = ", condest)
            ba_errors = {'bestapprox-'+k:v for k,v in ba_errors.items()}
            new_data = {'L':ref_lvl,'order':order,'contrast':contrast,'condest':condest} | errors | ba_errors
            df = pd.concat([df,pd.DataFrame(new_data,index=[0])],ignore_index=True)
            T = 'Adap' if TAdap else T 
            df.to_csv('../data/simplexExact_MultContrast_T{}.csv'.format(T),index=False)
            #df.to_csv('../data/test_differentstabs.csv'.format(order,well_posed),index=False)

            #Generate Data for plots (approximate solution)
            x_pts = np.array(plot_data_sol[0])
            ts = plot_data_sol[1]
            fun_y_pts = np.array(plot_data_sol[2][ int(len(ts)/2) ,: ])
            dt_y_pts = np.array(plot_data_sol[3][ int(len(ts)/2) ,: ])
            new_plot_data = {'x':x_pts,'y':fun_y_pts,'y_dt':dt_y_pts,'L':np.repeat(ref_lvl,len(x_pts)),'k':np.repeat(order,len(x_pts))}
            #import pdb; pdb.set_trace()
            plot_df = pd.concat([plot_df,pd.DataFrame.from_dict(new_plot_data)],ignore_index=True)
            plot_df.to_csv('../data/plots/simpleExact_T{0}_multipleContrasts_approx_plot_data_WP{1}_contrast{2}_k{3}.csv'.format(T,well_posed,contrast,order),index=False)

            #Save data for error plots (use np.genfromtxt("... .csv", delimiter=",")
            np.savetxt('../data/plots/simpleExact_T{0}_multipleContrasts_error_plot_xpts_WP{1}_k{2}_contrast{3}_ref{4}.csv'.format(T,well_posed,order,contrast,ref_lvl),plot_data_errors[0],delimiter=',')
            np.savetxt('../data/plots/simpleExact_T{0}_multipleContrasts_error_plot_ts_WP{1}_k{2}_contrast{3}_ref{4}.csv'.format(T,well_posed,order,contrast,ref_lvl),plot_data_errors[1],delimiter=',')
            np.savetxt('../data/plots/simpleExact_T{0}_multipleContrasts_error_plot_vals_WP{1}_k{2}_contrast{3}_ref{4}.csv'.format(T,well_posed,order,contrast,ref_lvl),plot_data_errors[2],delimiter=',')
            np.savetxt('../data/plots/simpleExact_T{0}_multipleContrasts_error_plot_dtvals_WP{1}_k{2}_contrast{3}_ref{4}.csv'.format(T,well_posed,order,contrast,ref_lvl),plot_data_errors[3],delimiter=',')

        


