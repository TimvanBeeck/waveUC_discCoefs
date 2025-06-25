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
from meshes import get_1Dmesh
import sys


if len(sys.argv) > 2:
    finalTime = float(sys.argv[1])
    contrast = float(sys.argv[2])
    print("Solving with final time T = {} and contrast {}".format(finalTime,contrast))
else: 
    raise ValueError('Please provide contrast')

ref_lvl = 4

#solver_type = "petsc-LU"  
solver_type = "pypardiso" # 
#solver_type = "direct" # 
well_posed = False #what does this do??

GCC = False #only controls wether data is set only on the lhs (False) or on both sides (True)

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


def csq_preInd(x,c1,c2,c3,pos,pos2):
    values = np.zeros(x.shape[1],dtype=ScalarType)
    pre_upper_coords = x[0] > pos 
    lower_coords = np.invert(pre_upper_coords)
    upper_coords = x[0] > pos2
    middle_coords = np.logical_and(x[0] >= pos, x[0] <= pos2)
    values[upper_coords] = np.full(sum(upper_coords), c3**2)
    values[middle_coords] = np.full(sum(middle_coords), c2**2)
    values[lower_coords] = np.full(sum(lower_coords), c1**2)
    print("values = ", values)
    return values


def gen_msh_pts(N,data_size,p1,p2):
    if N < 8:
        raise ValueError("N must be greater than 8")
    if data_size < 0.5:
        pts = [0.0,1.0,data_size,1-data_size,p1,p2]
        N = N + 1 - 6
    else: 
        pts = [0.0,1.0,data_size,p1,p2]
        N = N + 1 - 5
    print("Added initial points") 
    N_inner = int(N/3)
    N_outer = int((int(N/3)+1))
    for i in range(N_inner): 
        p_middle = p1 + (i+1)*(p2-p1)/(N_inner+1)
        if p_middle not in pts:
            pts.append(p_middle)
        else:
            pts.append(p_middle+0.01)
    for i in range(N_outer):
        p_left = (i+1)*(p1)/(N_outer+1)
        if p_left not in pts:
            pts.append(p_left)
        else: 
            pts.append(p_left+0.01)
        p_right = p2 + (i+1)*(1-p2)/(N_outer+1)
        if p_right not in pts:
            pts.append(p_right)
        else:
            pts.append(p_right+0.01)
    print("Added {} inner points".format(N_inner))
    print("Added {} outer points".format(2 * N_outer))
    return sorted(pts)

def gen_msh_pts_uniform(N,data_size,p1,p2):
    if N < 8:
        raise ValueError("N must be greater than 8")
    if data_size < 0.5:
        pts = [0.0,1.0,data_size,1-data_size,p1,p2]
        N = N + 1 - 6
    else: 
        pts = [0.0,data_size, 1.0,p1,p2]
        N = N + 1 - 5
    print("Added initial points") 
    for i in range(N): 
        p_middle = 0.0 + (i+1)*1.0/(N+1)
        for p in pts: 
            if abs(p_middle - p) < 1e-5:
                p_middle += 0.01
        pts.append(p_middle+0.01)
    return sorted(pts)
    

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

def Solve(ref_lvl,order,contrast,stabs,plot_sol=False,plot_error=False,T=0.5):
    t0 = 0

    data_size = 0.3
    N = 32 
    Nxs = [4,8,16,32,64]
    Nx = Nxs[ref_lvl]

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
    c3 = contrast
    
     #weights
    w1 = 3*np.pi
    w2 = c1*w1/c2

    pos= 0.4
    n_cos = 3 if contrast == 7.5 else 1 if contrast == 2.5 else 4
    pos2 = (2*np.pi*n_cos+w2*pos)/w2 


    #adapted to multiple jumps
    csq_Ind = lambda x : csq_preInd(x,c1,c2,c3,pos,pos2)



    # define quantities depending on space
    msh_pts = gen_msh_pts_uniform(Nx,data_size,pos,pos2)
    print(msh_pts)


    #msh = create_unit_interval(MPI.COMM_WORLD, Nx)
    msh = get_1Dmesh(msh_pts)

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


    def multiplejumps_sol(t,xu,c1,c2,w1,pos,n=3,dt=False):
        w2 = c1*w1/c2
        pos2 = (2*np.pi*n + w2*pos)/w2
        w3 = w1
        c3 = c1
        if not dt:
            u1 = ufl.cos(w1*c1*t) *ufl.cos(w1*(xu-pos))
            u2 = ufl.cos(w2*c2*t)*ufl.cos(w2*(xu-pos))
            u3 = ufl.cos(w3*c3*t)*ufl.cos(w3*(xu-pos2))
            sol = ufl.conditional(ufl.le(xu-pos2,0.0),ufl.conditional(ufl.le(xu-pos,0.0),u1,u2),u3)
        else: 
            u1 = -w1*c1*ufl.sin(w1*c1*t) *ufl.cos(w1*(xu-pos))
            u2 = -w2*c2*ufl.sin(w2*c2*t)*ufl.cos(w2*(xu-pos))
            u3 = -w3*c3*ufl.sin(w3*c3*t)*ufl.cos(w3*(xu-pos2))
            sol = ufl.conditional(ufl.le(xu-pos2,0.0),ufl.conditional(ufl.le(xu-pos,0.0),u1,u2),u3)
        return sol
    
    def sol(t,xu):
        return multiplejumps_sol(xu=xu[0],t=t,c1=c1,c2=c2,w1=w1,pos=pos,n=n_cos,dt=False)
    
    def dt_sol(t,xu):
        return multiplejumps_sol(xu=xu[0],t=t,c1=c1,c2=c2,w1=w1,pos=pos,n=n_cos,dt=True)
    


    st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind,stabs=stabs,sol=sol,dt_sol=dt_sol,well_posed=well_posed,data_dom_fitted=True,csq_Ind=csq_Ind)


    st.SetupSpaceTimeFEs()
    st.PreparePrecondGMRes()
    A_space_time_linop = st.GetSpaceTimeMatrixAsLinearOperator()
    b_rhs = st.GetSpaceTimeRhs()

    def SolveProblem(measure_errors = False,plot_sol=False,plot_error=False,only_plot=False):
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
            errors = st.MeasureErrorsRestrict(u_sol)
            initial_errors = st.MeasureInitialErrors(u_sol,restrict=True)
            errors = errors | initial_errors
            print("#----------------- Best approximation Errors-----------------#")
            ba_errors = st.MeasureErrorsRestrict(u_bestapprox)
            ba_initial_errors = st.MeasureInitialErrors(u_bestapprox,restrict=True)
            ba_errors = ba_errors | ba_initial_errors
        if plot_error: 
            plot_data_errors = st.PlotError(u_sol,N_space=100,N_time_subdiv=10,return_data=True)
        if only_plot: 
            return plot_data_sol
        elif plot_sol and plot_error:
            return errors, ba_errors,condest, plot_data_sol, plot_data_errors, iterations
        else: 
            return errors, ba_errors, condest, iterations 

    #cProfile.run('SolveProblem()')
    results = SolveProblem(measure_errors = False,plot_sol=plot_sol,plot_error=plot_error,only_plot=True) 

    print("Memory usage in (Gb) = ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6 )
    return results 

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


order = 3

T = finalTime

make_plots=True

plot_df = pd.DataFrame()


print("Solving with time T = ", T)

print("Order = {}, ref_lvl = {}, contrast = {}".format(order,ref_lvl,contrast))
stabs = set_stabs(c1=contrast)
results = Solve(ref_lvl=ref_lvl,order=order,contrast=contrast,stabs=stabs,plot_sol=make_plots,plot_error=make_plots,T=T)
plot_data_sol = results
#Generate Data for plots (approximate solution)
x_pts = np.array(plot_data_sol[0])
ts = plot_data_sol[1]
if T == 0.5:
    fun_y_pts = np.array(plot_data_sol[2][int(len(ts)-1) ,: ])
    dt_y_pts = np.array(plot_data_sol[3][int(len(ts)-1) ,: ])
else: 
    fun_y_pts = np.array(plot_data_sol[2][ int(len(ts)/2) ,: ])
    dt_y_pts = np.array(plot_data_sol[3][ int(len(ts)/2) ,: ])
new_plot_data = {'x':x_pts,'y':fun_y_pts,'y_dt':dt_y_pts,'L':np.repeat(ref_lvl,len(x_pts)),'k':np.repeat(order,len(x_pts)),'contrast':np.repeat(contrast,len(x_pts))}
plot_df = pd.concat([plot_df,pd.DataFrame.from_dict(new_plot_data)],ignore_index=True)
plot_df.to_csv('../dataNew/plots/MultipleJumpsExact_ApproxPlot_CompareT{}_contrast{}.csv'.format(T,contrast),index=False)

            
            



