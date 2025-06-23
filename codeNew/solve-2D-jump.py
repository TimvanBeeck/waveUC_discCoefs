import numpy as np
from math import pi,sqrt  
import ufl
from dolfinx import fem, io, mesh#, plot
from ufl import ds, dx, grad, inner, dS,jump,div
from mpi4py import MPI
from petsc4py.PETSc import ScalarType


from petsc4py import PETSc

from dolfinx.mesh import create_unit_interval
from dolfinx.io import XDMFFile
import sys
sys.setrecursionlimit(10**6)
from GMREs import GMRes
from space_time import * 
from precomp_time_int import theta_ref, d_theta_ref 
from meshes import get_2Dmesh_data_all_around
import pypardiso
import scipy.sparse as sp
import time
import cProfile
import resource

import pandas as pd 


#solver_type = "petsc-LU"  
solver_type = "pypardiso" # 
#solver_type = "direct" # 

well_posed = False

def GetLuSolver(msh,mat):
    solver = PETSc.KSP().create(msh.comm) 
    solver.setOperators(mat)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    return solver

# define alternative solvers here 
#def GetSpMat(mat):
#    ai, aj, av = mat.getValuesCSR()
#    Asp = sp.csr_matrix((av, aj, ai))
#    return Asp 


#class PySolver:
#    def __init__(self,Asp,psolver):
#        self.Asp = Asp
#        self.solver = psolver
#    def solve(self,b_inp,x_out):
#        x_py = self.solver.solve(self.Asp, b_inp.array )
#        x_out.array[:] = x_py[:]

class PySolver:
    def __init__(self,Asp,psolver):
        self.Asp = Asp
        self.solver = psolver
    def solve(self,b_inp,x_out): 
        self.solver._check_A(self.Asp)
        b = self.solver._check_b(self.Asp, b_inp.array)
        self.solver.set_phase(33)
        x_out.array[:] = self.solver._call_pardiso(self.Asp , b )[:]

#Generates a sequence of meshes
ls_mesh = get_2Dmesh_data_all_around(4,init_h_scale=5.0,data_size=0.25)

mesh_hierarchy = []
for Nx in [4,8,16,32]:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, Nx,Nx,mesh.CellType.quadrilateral)
    mesh_hierarchy.append(msh)

print("Mesh hierarchy created")

def Solve(ref_lvl,order,stabs,contrast=1.0):
    ref_lvl_to_N = [4,8,16,32]

    t0 = 0
    T = 0.75
    N = ref_lvl_to_N[ref_lvl-1]
    k = order
    q = order
    kstar = order
    qstar = order 
    #kstar = 1
    #if order == 1:
    #    qstar = 1
    #else:
    #    qstar = 0


    # define quantities depending on space
    #for j in range(len(ls_mesh)):
    #    with io.XDMFFile(ls_mesh[j].comm, "mesh-reflvl{0}.xdmf".format(j), "w") as xdmf:
    #        xdmf.write_mesh(ls_mesh[j])
    #msh = ls_mesh[ref_lvl]
    msh = mesh_hierarchy[ref_lvl-1]

    data_size = 0.25

    def omega_Ind_convex(x): 
        values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
        omega_coords = np.logical_or( ( x[0] <= data_size ), 
        np.logical_or(  ( x[0] >= 1-data_size ),        
            np.logical_or(   (x[1] >= 1-data_size ), (x[1] <= data_size)  )
            )
        )
        rest_coords = np.invert(omega_coords)
        values[omega_coords] = np.full(sum(omega_coords), 1.0)
        values[rest_coords] = np.full(sum(rest_coords), 0)
        return values
    """
    def sample_sol(t,xu):
        return ufl.cos(sqrt(2)*pi*t)*ufl.sin(pi*xu[0])*ufl.sin(pi*xu[1])

    def dt_sample_sol(t,xu):
        return -sqrt(2)*pi*ufl.sin(sqrt(2)*pi*t)*ufl.sin(pi*xu[0])*ufl.sin(pi*xu[1])
    """

    #Exact solution for discontinuous wave numbers (1D case)
    def ref_sol(x,t,p0,c1,c2,trunc,pos,dt=False):
        if not dt:
            #import pdb; pdb.set_trace()
            u1_plus = sum([((c2-c1)/(c2+c1))**k * (p0(k+x-c1*t)-p0(k-x-c1*t)) for k in range(trunc)])
            u1_minus = ((1/pos*c1)/(c2+c1))*sum([((c2-c1)/(c2+c1))**k * p0(c1/c2*(x-pos)+k+pos-c1*t) for k in range(trunc)])
        else:
            u1_plus = sum([((c2-c1)/(c2+c1))**k * (800*c1*(k+x-c1*t-1/5)*p0(k+x-c1*t)-800*c1*(k-x-c1*t-1/5)*p0(k-x-c1*t)) for k in range(trunc)])
            u1_minus = sum([((c2-c1)/(c2+c1))**k * 800*c1*(c1/c2*(x-pos)+k+pos-c1*t-1/5)*p0(c1/c2*(x-pos)+k+pos-c1*t) for k in range(trunc)])
        sol = ufl.conditional(ufl.le(x-pos,0),u1_plus,u1_minus)
        #sol = u1_plus
        return sol

    c1 = contrast
    c2 = 1.0
    trunc = 10
    pos = 0.5
    p0 = lambda x : 1/100*ufl.exp(-(20*(x-1/5))**2)

    def csq_preInd(x,c1,c2,pos):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        upper_coords = x[0] > pos 
        lower_coords = np.invert(upper_coords)
        values[upper_coords] = np.full(sum(upper_coords), c2**2)
        values[lower_coords] = np.full(sum(lower_coords), c1**2)
        return values

    csq_Ind = lambda x : csq_preInd(x,c1,c2,pos)

    """
    def sample_sol(t,xu):
        return ufl.cos(sqrt(2)*pi*t)*ufl.sin(pi*xu[0])*ufl.cos(pi*xu[1])

    def dt_sample_sol(t,xu):
        return -sqrt(2)*pi*ufl.sin(sqrt(2)*pi*t)*ufl.sin(pi*xu[0])*ufl.cos(pi*xu[1])
    
    
    def sample_sol(t,xu,alpha=1.0,beta=0.0):
        #alpha = -1/sqrt(2)
        #beta = -1/sqrt(2)
        return ref_sol(alpha*xu[0] + beta*xu[1],t,p0,c1,c2,trunc,pos,dt=False)

    def dt_sample_sol(t,xu,alpha=1.0,beta=0.0):
        #alpha = -1/sqrt(2)
        #beta = -1/sqrt(2)
        return ref_sol(alpha*xu[0] + beta*xu[1],t,p0,c1,c2,trunc,pos,dt=True)
    
    
    def sample_sol(t,xu,alpha=1.0,beta=0.0):
        #alpha = -1/sqrt(2)
        #beta = -1/sqrt(2)
        return ufl.sin(alpha*xu[0] + beta*xu[1]-t)

    def dt_sample_sol(t,xu,alpha=1.0,beta=0.0):
        #alpha = -1/sqrt(2)
        #beta = -1/sqrt(2)
        return -ufl.cos(alpha*xu[0] + beta*xu[1]-t)
    """

    #for testing purposes
    """
    def sample_sol(t,xu):
        u_minus = p0(xu[0] - t) - p0(-xu[0] - t)
        u_plus = p0(xu[0] - t)
        sol = ufl.conditional(ufl.le(xu[0]-0.5,0),u_plus,u_minus)
        return sol


    def dt_sample_sol(t,xu): #not correct, but only relevant for error 
        return 8*p0(xu[0] - t)*(xu[0] - t - 1/5)
    """

    def new_refsol(xu,t,c1,c2,w1,pos=0.5,dt=False):
        w2 = c1*w1/c2
        if not dt:
            u1 = ufl.cos(w1*c1*t) *ufl.cos(w1*(xu-pos))
            u2 = ufl.cos(w2*c2*t)*ufl.cos(w2*(xu-pos))
            sol = ufl.conditional(ufl.le(xu-pos,0.0),u1,u2)
        else: 
            u1 = -w1*c1*ufl.sin(w1*c1*t) *ufl.cos(w1*(xu-pos))
            u2 = -w2*c2*ufl.sin(w2*c2*t)*ufl.cos(w2*(xu-pos))
            sol = ufl.conditional(ufl.le(xu-pos,0.0),u1,u2)
        return sol
    
    def sample_sol(t,xu):
        return new_refsol(xu[0],t,c1=contrast,c2=1.0,w1=3*np.pi,pos=0.5,dt=False)

    def dt_sample_sol(t,xu):
        return new_refsol(xu[0],t,c1=contrast,c2=1.0,w1=3*np.pi,pos=0.5,dt=True) 
    
    st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind_convex,stabs=stabs,sol=sample_sol,dt_sol=dt_sample_sol,well_posed=well_posed,csq_Ind=csq_Ind)
    print("Space-time problem created")
    st.SetupSpaceTimeFEs()
    print("Space-time FEs created")
    st.PreparePrecondGMRes()
    print("Preconditioner prepared")
    A_space_time_linop = st.GetSpaceTimeMatrixAsLinearOperator()
    print("Space-time matrix created")
    b_rhs = st.GetSpaceTimeRhs()
    print("Space-time rhs created")

    # Prepare the solvers for problems on the slabs 
    #st.SetSolverSlab(GetLuSolver(st.msh,st.GetSlabMat())) # general slab
    #st.SetSolverFirstSlab(GetLuSolver(st.msh,st.GetSlabMatFirstSlab())) # first slab is special


    def SolveProblem(measure_errors = False):

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
            
            #st.SetSolverSlab( PySolver ( GetSpMat( st.GetSlabMat()))  )   # general slab
            #st.SetSolverFirstSlab( PySolver ( GetSpMat( st.GetSlabMatFirstSlab()) ) ) 
            #u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)
            u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching_improved,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)

            #also calculate best approximation??
            x_sol, _ = A_space_time_linop.createVecs() 
            st.compute_best_approx(x_sol)
            u_bestapprox = x_sol
        elif solver_type == "petsc-LU":
            st.SetSolverSlab(GetLuSolver(st.msh,st.GetSlabMat())) # general slab
            st.SetSolverFirstSlab(GetLuSolver(st.msh,st.GetSlabMatFirstSlab())) # first slab is special
            u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching_improved,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)
        else: 
            A_space_time = st.GetSpaceTimeMatrix() 
            u_sol,_ = A_space_time.createVecs() 
            solver_space_time = GetLuSolver(st.msh,A_space_time)
            solver_space_time.solve(u_sol ,b_rhs)
    
        end=time.time()
        print("elapsed time  " + str(end-start)+ " seconds")

    
        #st.PlotParaview(u_sol,time=np.linspace(0,T,25))
        st.PlotParaviewTS(u_sol)

        if measure_errors:
            errors = st.MeasureErrors(u_sol,dy_error=True)
            print("#----------------- Best approximation Errors-----------------#")
            ba_errors = st.MeasureErrors(u_bestapprox,dy_error=True)
            return errors, ba_errors
        

    #cProfile.run('SolveProblem()')
    errors, ba_errors = SolveProblem(measure_errors = True) 

    print("Memory usage in (Gb) = ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6 )
    return errors, ba_errors 

def set_stabs(c1,c2=1.0):
    stabs = {"data": 1e4, 
                "dual": 1e0,
                "primal": 1e-4,
                "primal-jump":1e-2, 
                "primal-jump-displ-grad":1e-2/max(c1**2,c2**2)**2, #same here
            } 
    return stabs

df = pd.DataFrame()


ref_lvls = [1,2,3,4]
orders = [2,3]
contrasts = [1.0,1.5,2.0,2.5]

#for testing 
ref_lvls = lambda order: [1,2,3] if order < 3 else [1,2] 
orders = [2,3]
contrasts = [1.0,1.5,2.0,2.5]

for order in orders: 
    for contrast in contrasts:
        stabs = set_stabs(contrast)
        for ref_lvl in ref_lvls(order):
            print('Order: {0}, Refinement level: {1}'.format(order,ref_lvl))
            errors, ba_errors = Solve(ref_lvl,order,stabs,contrast=contrast)
            ba_errors = {'bestapprox-'+k:v for k,v in ba_errors.items()}
            new_data = {'L':ref_lvl,'order':order,'contrast':contrast} | errors | ba_errors
            df = pd.concat([df,pd.DataFrame(new_data,index=[0])],ignore_index=True)
            df.to_csv('../dataNew/2D_errors_simpleRefsol_contrast{}.csv'.format(contrast),index=False)
