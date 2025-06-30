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

#solver_type = "petsc-LU"  
solver_type = "pypardiso" # 
#solver_type = "direct" # 
well_posed = False

GCC = True

## Helping Functions ##

def GetLuSolver(msh,mat):
    solver = PETSc.KSP().create(msh.comm) 
    solver.setOperators(mat)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    #solver.getPC().setFactorSolverType("mumps")
    return solver

class PySolver:
    def __init__(self,Asp,psolver):
        self.Asp = Asp
        self.solver = psolver
    def solve(self,b_inp,x_out): 
        self.solver._check_A(self.Asp)
        b = self.solver._check_b(self.Asp, b_inp.array)
        self.solver.set_phase(33)
        x_out.array[:] = self.solver._call_pardiso(self.Asp , b )[:]


def csq_preInd(x,c1,c2,pos):
    values = np.zeros(x.shape[1],dtype=ScalarType)
    upper_coords = x[0] > pos 
    lower_coords = np.invert(upper_coords)
    values[upper_coords] = np.full(sum(upper_coords), c2**2)
    values[lower_coords] = np.full(sum(lower_coords), c1**2)
    return values




#Exact solution for discontinuous wave numbers
def ref_sol(x,t,p0,c1,c2,trunc,pos,dt=False):
    if not dt:
        #import pdb; pdb.set_trace()
        u1_plus = sum([((c2-c1)/(c2+c1))**k * (p0(k+x[0]-c1*t)-p0(k-x[0]-c1*t)) for k in range(trunc)])
        u1_minus = ((1/pos*c1)/(c2+c1))*sum([((c2-c1)/(c2+c1))**k * p0(c1/c2*(x[0]-pos)+k+pos-c1*t) for k in range(trunc)])
    else:
        u1_plus = sum([((c2-c1)/(c2+c1))**k * (800*c1*(k+x[0]-c1*t-1/5)*p0(k+x[0]-c1*t)-800*c1*(k-x[0]-c1*t-1/5)*p0(k-x[0]-c1*t)) for k in range(trunc)])
        u1_minus = sum([((c2-c1)/(c2+c1))**k * 800*c1*(c1/c2*(x[0]-pos)+k+pos-c1*t-1/5)*p0(c1/c2*(x[0]-pos)+k+pos-c1*t) for k in range(trunc)])
    sol = ufl.conditional(ufl.le(x[0]-pos,0),u1_plus,u1_minus)
    #sol = u1_plus
    return sol

def Solve(ref_lvl,order,contrast,stabs,plot_sol=False,plot_error=False) :
    t0 = 0
    T = 1.0

    data_size = 0.25
    #ref_lvl_to_N = [1,2,4,8,16,32]
    #ref_lvl_to_N = [2,4,8,16,32,64]
    ref_lvl_to_N = [4,8,16,32,64,128]
    N = ref_lvl_to_N[ref_lvl]
    #N = 32
    #Nxs = [8,16,32,64,128,256]
    #Nxs = [16,32,64,128,256,512]

    Nxs = [4,8,16,32,64]
    Nx = Nxs[ref_lvl]
    #N_x = int(5*N)
    #N_x = int(2*int(N/2))
    #N_x = int(2*N)
    k = order
    q = order
    kstar = order
    qstar = order
    kstar = 1
    if order == 1:
        qstar = 1
    else:
        qstar = 0


    #Wave numbers and position of the jump
    c1 = contrast
    c2 = 1
    pos= 0.5 

    csq_Ind = lambda x : csq_preInd(x,c1,c2,pos)



    # define quantities depending on space
    msh = create_unit_interval(MPI.COMM_WORLD, Nx)

    #ToDo: modify this indicator function to a different domain... 
    def get_indicator_function(msh,Nx,GCC=True):
        if GCC:
            if Nx > 2:
                def omega_Ind(x): 
                    values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
                    #omega_coords = np.logical_or( ( x[0] <= 0.2 ), (x[0] >= 0.8 ))  
                    omega_coords = np.logical_or( ( x[0] <= data_size ), (x[0] >= 1-data_size ))  
                    rest_coords = np.invert(omega_coords)
                    values[omega_coords] = np.full(sum(omega_coords), 1.0)
                    values[rest_coords] = np.full(sum(rest_coords), 0)
                    #print("values = ", values)
                    return values

            else:
                x = ufl.SpatialCoordinate(msh)
                omega_indicator = ufl.Not(
                                    ufl.And(x[0] >= data_size, x[0] <= 1.0-data_size) 
                                        )
                omega_Ind = ufl.conditional(omega_indicator, 1, 0)

        else:
            if Nx > 2:
                def omega_Ind(x): 
                    values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
                    #omega_coords =  ( x[0] <= 0.2 )  
                    omega_coords =  ( x[0] <= data_size )  
                    #omega_coords =  ( x[0] <= 0.45 )  
                    #omega_coords =  ( x[0] <= 0.25 )  
                    rest_coords = np.invert(omega_coords)
                    values[omega_coords] = np.full(sum(omega_coords), 1.0)
                    values[rest_coords] = np.full(sum(rest_coords), 0)
                    #print("values = ", values)
                    return values
            else:
                x = ufl.SpatialCoordinate(msh)
                omega_indicator = ufl.And(x[0] <= data_size, x[0] >= 0.0) 
                omega_Ind = ufl.conditional(omega_indicator, 1, 0)
        
        return omega_Ind 

    omega_Ind = get_indicator_function(msh=msh,Nx=Nx,GCC=GCC)


    #Set up for exact solution
    p0 = lambda x : 1/100*ufl.exp(-(20*(x-1/5))**2)


    trunc = 10

    
    def sol(t,xu): 
        return ref_sol(xu,t,p0,c1,c2,trunc,pos,dt=False)

    def dt_sol(t,xu): 
        return ref_sol(xu,t,p0,c1,c2,trunc,pos,dt=True)

    st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind,stabs=stabs,sol=sol,dt_sol=dt_sol,well_posed=well_posed,data_dom_fitted= Nx > 2,csq_Ind=csq_Ind)


    st.SetupSpaceTimeFEs()
    st.PreparePrecondGMRes()
    A_space_time_linop = st.GetSpaceTimeMatrixAsLinearOperator()
    b_rhs = st.GetSpaceTimeRhs()

    def SolveProblem(measure_errors = False,plot_sol=False,plot_error=False):
        errors = {}
        start=time.time()
        if solver_type == "pypardiso":
            
            genreal_slab_solver = pypardiso.PyPardisoSolver()
            SlabMatSp = GetSpMat(st.GetSlabMat())
            
            genreal_slab_solver.factorize( SlabMatSp)   
            st.SetSolverSlab(PySolver(SlabMatSp, genreal_slab_solver))
            
            initial_slab_solver = pypardiso.PyPardisoSolver()
            SlabMatFirstSlabSp = GetSpMat( st.GetSlabMatFirstSlab())   
            initial_slab_solver.factorize( SlabMatFirstSlabSp  )   
            st.SetSolverFirstSlab(PySolver( SlabMatFirstSlabSp,  initial_slab_solver ))
            
            x_sweep_once  = fem.petsc.create_vector(st.SpaceTimeLfi)
            residual  = fem.petsc.create_vector(st.SpaceTimeLfi)
            diff = fem.petsc.create_vector(st.SpaceTimeLfi)
            st.pre_time_marching_improved(b_rhs, x_sweep_once)
            #st.Plot( x_sweep_once ,N_space=500,N_time_subdiv=20)
            u_sol,res, condest, iterations = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching_improved,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True,estimate_cond=True)
            

            #Calculate best approximation
            x_sol, _ = A_space_time_linop.createVecs() 
            st.compute_best_approx(x_sol)
            u_bestapprox = x_sol
            
            print(" x_sweep_once.array =",x_sweep_once.array )
            
            diff.array[:] = x_sweep_once.array[:] -  u_sol.array[:] 
        
            if plot_sol: 
                plot_data_sol = st.Plot( u_sol, N_space=500,N_time_subdiv=20,abs_val=False,return_data=True)
            
        elif solver_type == "petsc-LU":
            st.SetSolverSlab(GetLuSolver(st.msh,st.GetSlabMat())) # general slab
            st.SetSolverFirstSlab(GetLuSolver(st.msh,st.GetSlabMatFirstSlab())) # first slab is special
            u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching_improved,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)
        else: 
            A_space_time = st.GetSpaceTimeMatrix() 
            u_sol,_ = A_space_time.createVecs() 
            solver_space_time = GetLuSolver(st.msh,A_space_time)
            #solver_space_time.solve(u_sol ,b_rhs)
            solver_space_time.solve(b_rhs ,u_sol)
    
        end=time.time()
        print("elapsed time  " + str(end-start)+ " seconds")

        if measure_errors:
            errors = st.MeasureErrors(u_sol)
            initial_errors = st.MeasureInitialErrors(u_sol,restrict=False)
            errors = errors | initial_errors
            print("#----------------- Best approximation Errors-----------------#")
            ba_errors = st.MeasureErrors(u_bestapprox)
            ba_initial_errors = st.MeasureInitialErrors(u_bestapprox,restrict=False)
            ba_errors = ba_errors | ba_initial_errors
        if plot_error: 
            plot_data_errors = st.PlotError(u_sol,N_space=100,N_time_subdiv=10,return_data=True)
        
        if plot_sol and plot_error:
            return errors, ba_errors,condest, plot_data_sol, plot_data_errors, iterations
        else: 
            return errors, ba_errors, condest, iterations 

    #cProfile.run('SolveProblem()')
    results = SolveProblem(measure_errors = True,plot_sol=plot_sol,plot_error=plot_error) 

    print("Memory usage in (Gb) = ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6 )
    return results 




def set_stabs(c1,c2=1.0): 
    stabs = {#"data": 1e4,
                "data": 1e8,
                #"dual": 1.0,
                "dual": 1e0,
                #"primal": 1e-3,
                "primal": 1e-2,
                "primal-jump":1e0, # for simple sol: 1e-2, 
                #"primal-jump-vel":1.0,
                "primal-jump-displ-grad":1e0/max(c1**2,c2**2)**2 # for simple sol: 1e-2
            } 
    return stabs



df = pd.DataFrame()

#def ref_lvls(order):
#    if order < 3: 
#        return [1,2,3,4]
#    else: 
#        return [1,2,3,4]

def ref_lvls(order):
    return [1,2,3,4]    

#ref_lvls = [1,2,3]
orders = [2]
contrasts = [1.0,1.5,2.0]

make_plots=True

for order in orders:
    for contrast in contrasts: 
        plot_df = pd.DataFrame()
        for ref_lvl in ref_lvls(order):
            print("Order = {}, ref_lvl = {}, contrast = {}".format(order,ref_lvl,contrast))
            stabs = set_stabs(c1=contrast)
            results = Solve(ref_lvl=ref_lvl,order=order,contrast=contrast,stabs=stabs,plot_sol=make_plots,plot_error=make_plots)
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
            df.to_csv('../dataNew/weakDisc_evenhigherDt_noRestrict_1D_jumpingCoefs_k{0}.csv'.format(order,well_posed,contrast),index=False)
            #df.to_csv('../data/jumpCoefs/test_differentstabs.csv'.format(order,well_posed),index=False)

            #Generate Data for plots (approximate solution)
            x_pts = np.array(plot_data_sol[0])
            ts = plot_data_sol[1]
            fun_y_pts = np.array(plot_data_sol[2][ int(len(ts)/2) ,: ])
            dt_y_pts = np.array(plot_data_sol[3][ int(len(ts)/2) ,: ])
            new_plot_data = {'x':x_pts,'y':fun_y_pts,'y_dt':dt_y_pts,'L':np.repeat(ref_lvl,len(x_pts)),'k':np.repeat(order,len(x_pts))}
            #import pdb; pdb.set_trace()
            plot_df = pd.concat([plot_df,pd.DataFrame.from_dict(new_plot_data)],ignore_index=True)
            #plot_df.to_csv('../dataNew/plots/weakDisc_evenhigherDt_noRestrict_approx_plot_data_WP{0}_contrast{1}_k{2}.csv'.format(well_posed,contrast,order),index=False)

            #Save data for error plots (use np.genfromtxt("... .csv", delimiter=",")
            #np.savetxt('../dataNew/plots/weakDisc_evenhigherDt_noRestrict_error_plot_xpts_WP{0}_k{1}_contrast{2}_ref{3}.csv'.format(well_posed,order,contrast,ref_lvl),plot_data_errors[0],delimiter=',')
            #np.savetxt('../dataNew/plots/weakDisc_evenhigherDt_noRestrict_error_plot_ts_WP{0}_k{1}_contrast{2}_ref{3}.csv'.format(well_posed,order,contrast,ref_lvl),plot_data_errors[1],delimiter=',')
            #np.savetxt('../dataNew/plots/weakDisc_evenhigherDt_noRestrict_error_plot_vals_WP{0}_k{1}_contrast{2}_ref{3}.csv'.format(well_posed,order,contrast,ref_lvl),plot_data_errors[2],delimiter=',')
            #np.savetxt('../dataNew/plots/weakDisc_evenhigherDt_noRestrict_error_plot_dtvals_WP{0}_k{1}_contrast{2}_ref{3}.csv'.format(well_posed,order,contrast,ref_lvl),plot_data_errors[3],delimiter=',')

        


