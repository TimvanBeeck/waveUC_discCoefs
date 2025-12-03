from netgen.meshing import *
from netgen.csg import *
from netgen.meshing import Mesh as netmesh
from netgen.geom2d import SplineGeometry
from ngsolve.TensorProductTools import MeshingParameters
from ngsolve import *
from ngsolve.solvers import GMRes
from xfem import *
from xfem.lset_spacetime import *
import numpy as np
from ngsolve import Mesh as NGSMesh
from scipy.sparse import csr_matrix
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from math import pi 
import sys
sys.path.append("/home/janosch/projects/waveUC_discCoefs/code/ngsolve")
from space_time import space_time, SpaceTimeMat



tol = 1e-7 #GMRes
R0 = 0.25
R1 = 1.0 
R2 = sqrt(2)  
#R1_neg = 0.8
R1_neg = 0.8
R1_pos = 1.1*R1 
Rmid = R1 + 0.5*(R2-R1) 
n_mode = 50
#n_mode = 40
order_ODE = 1
#order_ODE = 4

c_minus = 1 
c_pos = 2.5 
maxh = 0.0625
print("maxh = ", maxh) 
bonus_intorder = 8
order = 1 
well_posed = True 

def get_order(order_global):
    
    k = order_global
    kstar = 1
    q = order_global
    if order_global == 1:
        qstar = 1
    else:
        qstar = 0

    return q,k,qstar,kstar



def Make1DMesh_givenpts(mesh_r_pts):

    mesh_1D = netmesh(dim=1)
    pids = []
    for r_coord in mesh_r_pts:
        pids.append (mesh_1D.Add (MeshPoint(Pnt(r_coord, 0, 0))))
    n_mesh = len(pids)-1
    for i in range(n_mesh):
        mesh_1D.Add(Element1D([pids[i],pids[i+1]],index=1))
    mesh_1D.Add (Element0D( pids[0], index=1))
    mesh_1D.Add (Element0D( pids[n_mesh], index=2))
    mesh_1D.SetBCName(0,"left")
    mesh_1D.SetBCName(1,"right")
    mesh_1D = NGSMesh(mesh_1D)
    return mesh_1D

def CreateAnnulusMesh(maxh=0.4,order_geom=5,domain_maxh=0.03,extra_refinement=False):
    
    geo = SplineGeometry()
    geo.AddCircle( (0,0), R0, leftdomain=0, rightdomain=1,bc="R0")

    geo.AddCircle( (0,0), R1_neg, leftdomain=1, rightdomain=2,bc="R1_neg")

    geo.AddCircle( (0,0), R1, leftdomain=2, rightdomain=3,bc="R1")

    geo.AddCircle( (0,0), R1_pos, leftdomain=3, rightdomain=4,bc="R1_pos")

    geo.AddCircle( (0,0), Rmid, leftdomain=4, rightdomain=5,bc="Rmid")
    geo.AddCircle( (0,0), R2, leftdomain=5, rightdomain=0,bc="R2")

    geo.SetMaterial(1, "B")
    geo.SetMaterial(2, "IF-inner")
    geo.SetMaterial(3, "IF-outer")

    geo.SetMaterial(4, "void")
    geo.SetMaterial(5, "omega-outer")
  
    if extra_refinement:
        geo.SetDomainMaxH(2, maxh/2)
        geo.SetDomainMaxH(3, maxh/2)

   
    mesh = NGSMesh(geo.GenerateMesh (maxh=maxh,quad_dominated=False))
    mesh.Curve(order_geom)
    return mesh



def P_DoFs(M,test,basis):
    return (M[test,:][:,basis])


def SolveModesRadial():
    npts = 100
    npts_red = int(npts/10)
    #mesh_r_pts = np.linspace(start=R0, stop=R1, num=npts, endpoint=False).tolist() 
    #mesh_r_pts += np.linspace(start=R1, stop=R2, num=npts, endpoint=True).tolist() 

    mesh_r_pts = np.linspace(start=R0, stop=R1_neg, num=npts, endpoint=False).tolist() 
    mesh_r_pts += np.linspace(start=R1_neg, stop=R1, num=npts, endpoint=False).tolist() 
    mesh_r_pts += np.linspace(start=R1, stop=R1_pos, num=npts, endpoint=False).tolist() 
    mesh_r_pts += np.linspace(start=R1_pos, stop=R2, num=npts_red, endpoint=True).tolist() 

    #mesh_r_pts = np.linspace(start=0.0, stop=pi, num=npts, endpoint=True).tolist() 
    print("mesh_r_pts  = ", mesh_r_pts) 

    npts_eval = 1000
    npts_eval_red = int(npts_eval/50)
    npts_eval_less = int(npts_eval/10)
 
    #eval_r_pts = np.linspace(start=R0, stop=R1, num=npts_eval, endpoint=False).tolist() 
    #eval_r_pts += np.linspace(start=R1, stop=R2, num=npts_eval, endpoint=True).tolist() 

    eval_r_pts = np.linspace(start=R0, stop=R1_neg,  num= npts_eval_less , endpoint=False).tolist() 
    eval_r_pts += np.linspace(start=R1_neg, stop=R1, num= npts_eval , endpoint=False).tolist() 
    eval_r_pts += np.linspace(start=R1, stop=R1_pos, num= npts_eval , endpoint=False).tolist() 
    eval_r_pts += np.linspace(start=R1_pos, stop=R2, num= npts_eval_red , endpoint=True).tolist() 

    mesh_1D = Make1DMesh_givenpts(mesh_r_pts)  


    c_disc = IfPos( R1 - x, c_minus, c_pos )
    #eval_pts = np.array( [ c_disc( mesh_1D(xpt) )  for xpt in  mesh_r_pts ] ) 
    #plt.plot(mesh_r_pts, eval_pts) 
    #plt.show() 


    fes = H1(mesh_1D, complex=True,  order=order_ODE, dirichlet="left|right")
    #print("fes.FreeDofs() = " , fes.FreeDofs())
    freedofs = [i for i in range( len(fes.FreeDofs() )) if  fes.FreeDofs()[i]  ] 
    #print("freedofs = ", freedofs )
    #input("")
    #fes = H1(mesh_1D, complex=True,  order=order_ODE, dirichlet=[])
    u,v = fes.TnT()

    gfu = GridFunction(fes)

    m_mass = BilinearForm (fes , symmetric=False, check_unused=False)
    m_mass +=  (1.0 / n_mode**2) * u * v * dx
    m_mass.Assemble()

    k_diff = BilinearForm (fes, symmetric=False, check_unused=False)
    #k_diff +=  (1.0 / n_mode**2) * grad(u) * grad(v)  * dx 
    k_diff +=  (c_disc / n_mode**2) * grad(u) * grad(v)  * dx 
    k_diff +=  c_disc * (-1) * (1.0 / n_mode**2) * (1.0/x) * grad(u) * v * dx 
    k_diff +=  (c_disc/x**2) * u * v * dx    
    k_diff.Assemble()

    rows_m,cols_m,vals_m = m_mass.mat.COO()
    #M_if = P_DoFs(csr_matrix((vals_m,(rows_m,cols_m))),decomp.IF2dof[if_idx],decomp.IF2dof[if_idx]).todense()
    M_mat =  P_DoFs( csr_matrix((vals_m,(rows_m,cols_m))), freedofs, freedofs ).todense()

    rows_k,cols_k,vals_k = k_diff.mat.COO()
    K_mat =  P_DoFs( csr_matrix((vals_k,(rows_k,cols_k))) , freedofs, freedofs ).todense()
    #K_if = P_DoFs(csr_matrix((vals_k,(rows_k,cols_k))),decomp.IF2dof[if_idx],decomp.IF2dof[if_idx]).todense()

    lam_disc,lam_vec =  scipy.linalg.eig(K_mat, b=M_mat, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True)


    # sorting eigenpairs
    idx_sorted = np.argsort(lam_disc)
    lam_disc = lam_disc[idx_sorted]
    lam_vec = lam_vec[:,idx_sorted]

    print("lam_disc = ", lam_disc)
    input("")

    lam_diag = np.diag(lam_disc)
    lam_vec = lam_vec @ np.diag(  1/np.sqrt(np.diag( lam_vec.conj().T @ (M_mat @ lam_vec))) )

    idx_modes = 1 
        
    '''     
    for i in range(10):
        print("mode nr = {0} ".format(i))
        gfu.vec.data.FV().NumPy()[:] = 0.0
        gfu.vec.data.FV().NumPy()[freedofs]  = lam_vec[:,i] 
        eval_pts = np.array( [ gfu( mesh_1D(xpt) )  for xpt in  eval_r_pts  ] ) 

        plt.plot(eval_r_pts  , eval_pts) 
        #plt.plot(range(len(lam_vec[:,0])), lam_vec[:,0] ) 
        plt.show()
    '''
    gfu.vec.data.FV().NumPy()[:] = 0.0
    gfu.vec.data.FV().NumPy()[freedofs]  = lam_vec[:,idx_modes] 
    eval_vals = np.real( np.array( [   gfu( mesh_1D(xpt) )  for xpt in  eval_r_pts  ] )  )

    return eval_r_pts, eval_vals.tolist(), lam_disc[idx_modes]  

eval_r_pts, eval_vals, lami = SolveModesRadial()
#print(" eval_r_pts  = ", eval_r_pts)
#print(" ") 
print(" eval_vals  = ", eval_vals )


spline_order = 1 # order of Spline approx. for coefficients

rs = [ eval_r_pts[0] for i in range(spline_order)] + eval_r_pts + [ eval_r_pts[-1] for i in range(spline_order)]
c_clean = [ eval_vals[0] for i in range(spline_order)] + eval_vals  + [ eval_vals[-1] for i in range(spline_order)]

r = sqrt(x*x + y*y) 
rho_cleanB = BSpline(spline_order,rs, c_clean )(r)
#cBB = BSpline(spline_order,r,c_clean)
#rhoBB =BSpline(spline_order,r,rho_clean)
#eval_c = [cBB(pp) for pp in rS[::-1]]
#eval_c[-1] = cBB(rS[::-1][-1]-1e-11)
#eval_rho = [rhoBB(pp) for pp in rS[::-1]]

#mesh = CreateAnnulusMesh(maxh=maxh, extra_refinement=True ) 
mesh = CreateAnnulusMesh(maxh=maxh, extra_refinement=False ) 
Draw(mesh)
Draw(rho_cleanB, mesh, 'cr')  

rho_B_vals = [  rho_cleanB( mesh(xr,0.0) )  for xr in  eval_r_pts ] 

#plt.plot( eval_r_pts, eval_vals, label='exact' ) 
#plt.plot( eval_r_pts, rho_B_vals,label='Bspline'  ) 
#plt.show() 

diff = [ abs( eval_vals[i] -  rho_B_vals[i] )  for i in range(len(rho_B_vals))  ] 
#plt.loglog( eval_r_pts[1:-2], diff[1:-2])
#plt.show()
print("diff = ", diff)

theta = atan2(y,x)
#angular_mode = exp(1j*theta) 
angular_mode = cos( n_mode *  theta) 
Draw(angular_mode, mesh, 'angular')
wp_mode_space = rho_cleanB *  angular_mode 
Draw(wp_mode_space , mesh, 'wp-mode')



domain_values = {'B': c_minus,  'IF-inner': c_minus,  'IF-outer': c_pos, 'void' : c_pos,  'omega-outer' : c_pos  }
#domain_values = {'inner': 3.7,  'outer': 1}


c_disc = mesh.MaterialCF(domain_values)
#help(c_disc)
Draw(c_disc, mesh, 'cdisk')

domain_values_2 = { 'B': 1.0,  'IF-inner': 1.0,  'IF-outer': 1.0, 'void' : 1.0,  'omega-outer' : 1.0 }
c_squared = mesh.MaterialCF(domain_values_2)


def CheckSpatial(c_disc,lami, wp_mode_space,mesh): 

    fes = H1(mesh, order=order, dirichlet="R2")

    # define trial- and test-functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # the right hand side
    f = LinearForm(fes)
    f +=  lami.real *  wp_mode_space  * v * dx(bonus_intorder =  bonus_intorder) 

    # the bilinear-form
    a = BilinearForm(fes, symmetric=True)
    a +=  c_disc * grad(u)*grad(v)*dx

    a.Assemble()
    f.Assemble()

    # the solution field
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
    Draw(gfu,mesh,'gfu')
    Draw(sqrt( (gfu- wp_mode_space)*(gfu-wp_mode_space)), mesh, 'err')
    l2_abserr =  sqrt (Integrate ( (gfu- wp_mode_space)*(gfu-wp_mode_space), mesh, order =  2*max(order,4) +  bonus_intorder ))
    l2_norm = sqrt (Integrate (  wp_mode_space*wp_mode_space , mesh, order =  2*max(order,4) +  bonus_intorder  ))
    print ("relative L2-error:",  l2_abserr /  l2_norm   )

#CheckSpatial(c_disc,lami, wp_mode_space,mesh) 

#l2_errors = [ 0.23088207463877092, 0.06298057321067921]

stabs = {"data": 1e4,
         "dual": 1,
         "primal": 1e-4,
         "primal-jump":1e1,
         "primal-jump-displ-grad":1e1,
         "Tikh": 1e-18
        }

def SolveProblem( order_global, lami, wp_mode_space ):

    q,k,qstar,kstar = get_order(order_global)
    time_order = 2*max(q,qstar)

    N = 32
    tstart = 0.0
    tend = 2.0
    delta_t = tend / N
    # Level-set functions specifying the geoemtry
    told = Parameter(tstart)
    t = told + delta_t * tref

    # define exact solution
    qpi = pi/4 
    t_slice = [ tstart  + n*delta_t + delta_t*tref for n in range(N)]
    #u_exact_slice = [ cos(  sqrt(lami) * t_slice[n] ) * wp_mode_space  for n in range(N)]
    #ut_exact_slice = [ sqrt(lami) * (-1) * sin(  sqrt(lami) * t_slice[n] ) * wp_mode_space for n in range(N)]

    qpi = pi/4 
    m_sol = 2
    u_exact_slice = [  5*cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]
    ut_exact_slice = [ -5 * sqrt(2) * m_sol * qpi * sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]
    

    st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=tend,delta_t=delta_t,mesh=mesh,stabs=stabs,
                    t_slice=t_slice, u_exact_slice=u_exact_slice, ut_exact_slice=ut_exact_slice, tstart=tstart, 
                    told=told, well_posed = well_posed, c_squared = c_squared) 

    st.SetupSpaceTimeFEs()
    st.SetupRightHandSide()
    st.PreparePrecondGMRes()
    
    A_linop = SpaceTimeMat(ndof = st.X.ndof,mult = st.ApplySpaceTimeMat)
    PreTM = SpaceTimeMat(ndof = st.X.ndof,mult = st.TimeMarching)

    st.gfuX.vec.data = GMRes(A_linop, st.f.vec, pre=PreTM, maxsteps = 10000, tol = tol,
                  callback=None, restart=None, startiteration=0, printrates=True)

    l2_errors_Q = st.MeasureErrors(st.gfuX )
    print("Errors in Q (all) = ", l2_errors_Q)

    '''
    if plotting:
        # Plotting
        fesp1 = H1(mesh, order=1, dgjumps=True)
        # Time finite element (nodal!)
        tfe = ScalarTimeFE(st.q)
        # Space-time finite element space
        st_fes = tfe * fesp1
        lset_p1 = GridFunction(st_fes)
        lset_adap_st = LevelSetMeshAdaptation_Spacetime(mesh, order_space=st.k,
                                                        order_time=st.q,
                                                        threshold=0.1,
                                                        discontinuous_qn=True)
        ci = CutInfo(mesh, time_order=0)

        uh_slab = GridFunction(st.W_slice_primal)
        u_slab = GridFunction(st.W_slice_primal)
        diff_slab = GridFunction(st.W_slice_primal)
        u_slab_node = GridFunction(st.V_space)

        vtk_out = [B_lset , lset_Omega, data_lset, Q_lset,u_slab, uh_slab, lset_p1, diff_slab ]
        vtk_out_names = ["B", "Omega", "data", "Q","u", "uh","lsetp1","diff"]

        vtk = SpaceTimeVTKOutput(ma=mesh, coefs=vtk_out, names=vtk_out_names,
                                 filename="2D-cylinder-reflvl{0}-q{1}".format(ref_lvl,q), subdivision_x=3,
                                 subdivision_t=3)
        print("ploting ...")
        n = 0 
        told.Set(st.tstart)
        while tend - told.Get() > st.delta_t / 2:
            #print("n = ", n) 
            SpaceTimeInterpolateToP1(levelset, tref, lset_p1)
            dfm = lset_adap_st.CalcDeformation(levelset, tref)
            ci.Update(lset_p1, time_order=0)
            uh_slab.vec.FV().NumPy()[:] = st.gfuX.components[0].components[n].vec.FV().NumPy()[:]
            
            times = [xi for xi in st.W_slice_primal.TimeFE_nodes()]
            for i,ti in enumerate(times):
                u_slab_node.Set(fix_tref(st.u_exact_slice[n],ti ))
                u_slab.vec[i*st.V_space.ndof : (i+1)*st.V_space.ndof].data = u_slab_node.vec[:]
            diff_slab.vec.FV().NumPy()[:] = u_slab.vec.FV().NumPy()[:] - uh_slab.vec.FV().NumPy()[:]

            vtk.Do(t_start=told.Get(), t_end=told.Get() + st.delta_t)
            told.Set(told.Get() + st.delta_t)
            n += 1 
    '''


    return delta_t, l2_errors_Q  
 

if True:

    for order_global in [1]:
        
        errors = { "delta-t": [], 
                   "B" : [ ],
                  "B-complement": [],
                  "omega" : [ ],  
                  "Q_all" : [ ] 
                  } 
 
        result = SolveProblem(order_global, lami, wp_mode_space )
        

# smooth sol as data, no jump
# N = [8,16,32, 64 ] 
# maxh = [0.25,0.125, 0.0625, 0.03125  ]
# l2_errors = [0.4750260066626608, 0.09060563861975547, 0.014688240968942253 ] 
# with well_posed = True 
# l2_errors = [ 0.3400832328035614 ,0.06398742850314416, 0.014509269239931124 ]  

