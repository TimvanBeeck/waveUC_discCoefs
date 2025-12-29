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
import scipy.interpolate as si
import sys
sys.path.append("/home/janosch/projects/waveUC_discCoefs/code/ngsolve")
from space_time import space_time, SpaceTimeMat


if ( len(sys.argv) > 1 and int(sys.argv[1]) in [1,10]  ):
    n_mode = int(sys.argv[1])
else:
    raise ValueError('Ivalid input')

if ( len(sys.argv) > 2):
    if int(sys.argv[2]) == 0:
        well_posed = False
    else:
        well_posed = True

#n_mode = 1

print("n_mode = ", n_mode)
print("well_posed = ", well_posed)


tol = 1e-7 #GMRes
tol = 1e-4 #GMRes

#R0 = 0.25
R0 = 0.6

R1 = 1.0 
#R2 = 1.75
#R2 = sqrt(2)  
R2 = 1.5 
#R1_neg = 0.8
R1_neg = 0.8
R1_pos = 1.1*R1 
Rmid = R1 + 0.5*(R2-R1) 

Rb = 0.85
Rsep = 1.3

print("Rmid = ", Rmid)
#n_mode = 10 # ref

#n_mode = 1
#n_mode = 20 # see later if that can be resolved
#order_ODE = 1
order_ODE = 4

c_minus = 1 
#c_pos = 2.5 
c_pos = 20.0 
#c_pos = 1.0
maxh =  0.25
#maxh =  0.4
print("maxh = ", maxh) 
bonus_intorder = 8
#bonus_intorder = 16

order = 2 
order_geom = order
problem_type = "ill-posed"
if well_posed:
    problem_type = "well-posed"
check_spatial_conv = False 


def get_order(order_global):
    
    k = order_global
    kstar = k
    q = order_global
    if order_global == 1:
        qstar = 1
    else:
        qstar = q

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

def MakeSepMesh(ra,rx,rb,npts):

    left_pts = np.linspace(start=ra, stop=rx, num=npts, endpoint=False).tolist() 
    right_pts = np.linspace(start=rx, stop=rb, num=npts, endpoint=True).tolist() 
    
    all_pts = left_pts + right_pts 

    #print("left_pts = ", left_pts)
    #print(" ")
    #print("right_pts = ", right_pts) 
     
    mesh_1D = netmesh(dim=1)

    pids = []
    for r_coord in all_pts:
        pids.append (mesh_1D.Add (MeshPoint(Pnt(r_coord, 0, 0))))
    n_mesh = len(pids)-1
    for i in range(n_mesh):
        idx = 1 
        if i >= len(left_pts):
            idx = 2 
            #print("pt = ",  all_pts[i])
        mesh_1D.Add(Element1D([pids[i],pids[i+1]],index=idx))
    mesh_1D.Add (Element0D( pids[0], index=1))
    mesh_1D.Add (Element0D( pids[-1], index=1))
    
    #pids = []
    #for r_coord in right_pts:
    #    pids.append (mesh_1D.Add (MeshPoint(Pnt(r_coord, 0, 0))))
    #n_mesh = len(right_pts)-1
    #for i in range(n_mesh):
    #    mesh_1D.Add(Element1D([pids[i],pids[i+1]],index=1))

    mesh_1D.SetMaterial(1, "leftdom")
    mesh_1D.SetMaterial(2, "rightdom")

    mesh_1D.SetBCName(0,"left")
    mesh_1D.SetBCName(1,"right")
    mesh_1D = NGSMesh(mesh_1D)
    return mesh_1D



def CreateAnnulusMesh(maxh=0.4,order_geom=1,domain_maxh=0.03,extra_refinement=False):
    
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
        #geo.SetDomainMaxH(2, maxh/2)
        #geo.SetDomainMaxH(3, maxh/2)
        geo.SetDomainMaxH(2, maxh/2)
        geo.SetDomainMaxH(3, maxh/2)

   
    mesh = NGSMesh(geo.GenerateMesh (maxh=maxh,quad_dominated=False))
    #mesh.Curve(order_geom)
    return mesh



def CreateAnnulusMesh2(maxh=0.4,order_geom=1,domain_maxh=0.03,extra_refinement=False):
    
    geo = SplineGeometry()
    geo.AddCircle( (0,0), R0, leftdomain=0, rightdomain=1,bc="R0")
    #geo.AddCircle( (0,0), R0, leftdomain=1, rightdomain=0,bc="R0")

    #geo.AddCircle( (0,0), 1.0, leftdomain=1, rightdomain=2,bc="R1")
    
    #geo.AddCircle( (0,0), R2, leftdomain=2, rightdomain=0,bc="R2")

    #geo.SetMaterial(1, "B")
    #geo.SetMaterial(2, "IF-inner")
    #geo.SetMaterial(3, "omega-outer")
  
    geo.AddCircle( (0,0), R2, leftdomain=1, rightdomain=0,bc="R2")
   
    mesh = NGSMesh(geo.GenerateMesh (maxh=maxh,quad_dominated=False))
    mesh.Curve(order_geom)
    return mesh


def CreateAnnulusMesh3(maxh=0.4,order_geom=1,domain_maxh=0.03,extra_refinement=False):
    
    #Rb = 0.85
    #Rsep = 1.3

    geo = SplineGeometry()
    geo.AddCircle( (0,0), R0, leftdomain=0, rightdomain=1,bc="R0")
    geo.AddCircle( (0,0), Rb, leftdomain=1, rightdomain=2,bc="Rb")
    geo.AddCircle( (0,0), R1, leftdomain=2, rightdomain=3,bc="R1")
    geo.AddCircle( (0,0), Rsep, leftdomain=3, rightdomain=4,bc="Rsep")
    geo.AddCircle( (0,0), R2, leftdomain=4, rightdomain=0,bc="R2")

    geo.SetMaterial(1, "B")
    geo.SetMaterial(2, "IF-inner")
    geo.SetMaterial(3, "void")
    geo.SetMaterial(4, "omega-outer")
  
    if extra_refinement:
        geo.SetDomainMaxH(2, maxh/2)
        geo.SetDomainMaxH(3, maxh/2)
        #geo.SetDomainMaxH(1, maxh/2)

   
    mesh = NGSMesh(geo.GenerateMesh (maxh=maxh,quad_dominated=False))
    mesh.Curve(order_geom)
    #mesh.Curve(1)
    return mesh



def P_DoFs(M,test,basis):
    return (M[test,:][:,basis])


def SolveModesRadial():
    npts = 100
    #npts = 140
    npts_red = int(npts/10)
    #mesh_r_pts = np.linspace(start=R0, stop=R1, num=npts, endpoint=False).tolist() 
    #mesh_r_pts += np.linspace(start=R1, stop=R2, num=npts, endpoint=True).tolist() 

    #mesh_r_pts = np.linspace(start=R0, stop=R1_neg, num=npts, endpoint=False).tolist() 
    #mesh_r_pts += np.linspace(start=R1_neg, stop=R1, num=npts, endpoint=False).tolist() 
    #mesh_r_pts += np.linspace(start=R1, stop=R1_pos, num=npts, endpoint=False).tolist() 
    #mesh_r_pts += np.linspace(start=R1_pos, stop=R2, num=npts_red, endpoint=True).tolist() 

    #Rb = 0.85
    #Rsep = 1.3

    #mesh_r_pts = np.linspace(start=R0, stop=Rb, num=npts, endpoint=False).tolist() 
    mesh_r_pts = np.linspace(start=R0, stop=Rb, num=npts_red, endpoint=False).tolist() 
    mesh_r_pts += np.linspace(start=Rb, stop=R1, num=npts, endpoint=False).tolist() 
    mesh_r_pts += np.linspace(start=R1, stop=Rsep, num=npts, endpoint=False).tolist() 
    mesh_r_pts += np.linspace(start=Rsep, stop=R2, num=npts_red, endpoint=True).tolist() 


    #mesh_r_pts = np.linspace(start=0.0, stop=pi, num=npts, endpoint=True).tolist() 
    #print("mesh_r_pts  = ", mesh_r_pts) 

    #npts_eval = 50
    npts_eval = 5000
    npts_eval_red = int(npts_eval/50)
    npts_eval_less = int(npts_eval/10)
 
    #eval_r_pts = np.linspace(start=R0, stop=R1, num=npts_eval, endpoint=False).tolist() 
    #eval_r_pts += np.linspace(start=R1, stop=R2, num=npts_eval, endpoint=True).tolist() 

    #eval_r_pts = np.linspace(start=R0, stop=R1_neg,  num= npts_eval_less , endpoint=False).tolist() 
    #eval_r_pts += np.linspace(start=R1_neg, stop=R1, num= npts_eval , endpoint=False).tolist() 
    #eval_r_pts += np.linspace(start=R1, stop=R1_pos, num= npts_eval , endpoint=False).tolist() 
    #eval_r_pts += np.linspace(start=R1_pos, stop=R2, num= npts_eval_red , endpoint=True).tolist() 

    #eval_r_pts = np.linspace(start=R0, stop=Rb,  num= npts_eval_less , endpoint=False).tolist() 
    #eval_r_pts += np.linspace(start=Rb, stop=R1, num= npts_eval , endpoint=False).tolist() 
    #eval_r_pts += np.linspace(start=R1, stop=Rsep, num= npts_eval , endpoint=False).tolist() 
    #eval_r_pts += np.linspace(start=Rsep, stop=R2, num= npts_eval_red , endpoint=True).tolist() 

    eval_r_pts = np.linspace(start=R0, stop=R1,  num= npts_eval , endpoint=False).tolist() 
    eval_r_pts += np.linspace(start=R1, stop=R2, num= npts_eval , endpoint=True).tolist() 

    #mesh_1D = Make1DMesh_givenpts(mesh_r_pts)  
     
    mesh_1D =  MakeSepMesh(R0,R1,R2,npts)
    print("mesh_1D.GetMaterials() = ", mesh_1D.GetMaterials())
    #c_disc = IfPos( R1 - x, c_minus, c_pos )
    dvals = {'leftdom': c_minus, 'rightdom' : c_pos  }
    
    c_disc = mesh_1D.MaterialCF(dvals)
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

    #print("lam_disc = ", lam_disc)
    #print("sqrt(lam_disc) = ", np.sqrt(lam_disc.real)) 
    #input("")

    lam_diag = np.diag(lam_disc)
    lam_vec = lam_vec @ np.diag(  1/np.sqrt(np.diag( lam_vec.conj().T @ (M_mat @ lam_vec))) )
    
    
    idx_modes = 1 
    #if n_mode == 1:
    #    idx_modes = 2
           
    '''
    for i in range(10):
        print("mode nr = {0} ".format(i))
        gfu.vec.data.FV().NumPy()[:] = 0.0
        gfu.vec.data.FV().NumPy()[freedofs]  = lam_vec[:,i] 
        eval_pts = np.array( [ gfu( mesh_1D(xpt) )  for xpt in  eval_r_pts  ] ) 

        plt.plot(eval_r_pts  , eval_pts) 
        #plt.plot(range(len(lam_vec[:,0])), lam_vec[:,0] ) 
        plt.show()
         
        ax = plt.gca()
        ymin = 1e-4
        ymax = 1e3
        ax.set_ylim([ymin, ymax])
        plt.semilogy(eval_r_pts  , np.abs( eval_pts) ) 
        plt.show()
    '''

    #print("eval_r_pts  = ", eval_r_pts) 
    gfu.vec.data.FV().NumPy()[:] = 0.0
    gfu.vec.data.FV().NumPy()[freedofs]  = lam_vec[:,idx_modes] 
    eval_vals = np.real( np.array( [   gfu( mesh_1D(xpt) )  for xpt in  eval_r_pts  ] )  )
    

    return eval_r_pts, eval_vals.tolist(), lam_disc[idx_modes]  


eval_r_pts, eval_vals, lami = SolveModesRadial()

if well_posed: 
    name_str_r = "whispering-gallery-mode-nr" + "{0}".format(n_mode)  + "-radial.dat"
    abs_mode = np.abs( eval_vals) 
    results_r = [ np.array( eval_r_pts, dtype=float) , np.array( eval_vals , dtype=float) , np.array( abs_mode / np.max(abs_mode) , dtype=float) ]
    header_str_r = "r val absval"
    np.savetxt(fname ="data/{0}".format(name_str_r),
                       X = np.transpose(results_r),
                       header = header_str_r,
                       comments = '')
#input("")

#print(" eval_r_pts  = ", eval_r_pts)
#print(" ") 
#print(" eval_vals  = ", eval_vals )


spline_order = 2 # order of Spline approx. for coefficients

#rs = [ eval_r_pts[0] for i in range(spline_order)] + eval_r_pts + [ eval_r_pts[-1] for i in range(spline_order)]
#c_clean = [ eval_vals[0] for i in range(spline_order)] + eval_vals  + [ eval_vals[-1] for i in range(spline_order)]

r = sqrt(x*x + y*y) 

tck = si.splrep(eval_r_pts,  eval_vals  , k=spline_order )  # Get knots, coefficients, and degree

# Let scipy generate the BSpline 
# see https://forum.ngsolve.org/t/b-spline-use/3007
knots = tck[0]  # Knot vector
vals = tck[1]   # Control points (called "vals" in NGSolve)
sporder = tck[2] + 1 # Spline order corrected by 1

print("sporder = ", sporder)
# Pass the extracted parameters to an NGSolve BSpline
ngspline = BSpline(sporder, list(knots), list(vals))
rho_cleanB = BSpline(sporder, list(knots), list(vals) )(r)




#cBB = BSpline(spline_order,r,c_clean)
#rhoBB =BSpline(spline_order,r,rho_clean)
#eval_c = [cBB(pp) for pp in rS[::-1]]
#eval_c[-1] = cBB(rS[::-1][-1]-1e-11)
#eval_rho = [rhoBB(pp) for pp in rS[::-1]]

#mesh = CreateAnnulusMesh(maxh=maxh, extra_refinement=True ) 
#mesh = CreateAnnulusMesh(maxh=maxh, order_geom = order, extra_refinement=False) 





#Draw(mesh)
#Draw(rho_cleanB, mesh, 'cr')  

#rho_B_vals = [  rho_cleanB( mesh(xr,0.0) )  for xr in  eval_r_pts ] 

#plt.plot( eval_r_pts, eval_vals, label='exact' ) 
#plt.plot( eval_r_pts, rho_B_vals,label='Bspline'  ) 
#plt.show() 

#diff = [ abs( eval_vals[i] -  rho_B_vals[i] )  for i in range(len(rho_B_vals))  ] 
#plt.loglog( eval_r_pts[1:-2], diff[1:-2])
#plt.show()
#print("diff = ", diff)

theta = atan2(y,x)
#angular_mode = exp(1j*theta) 
angular_mode = cos( n_mode *  theta) 
#Draw(angular_mode, mesh, 'angular')
wp_mode_space = rho_cleanB *  angular_mode 

#Draw(wp_mode_space , mesh, 'wp-mode')
#input("")

#domain_values = {'B': c_minus,  'IF-inner': c_minus,  'IF-outer': c_pos, 'void' : c_pos,  'omega-outer' : c_pos  }
#domain_values = {'B': c_minus,  'IF-inner': c_minus,  'IF-outer': c_pos, 'void' : c_pos,  'omega-outer' : c_pos  }
domain_values = {'B': c_minus,  'IF-inner': c_minus,   'void' : c_pos,  'omega-outer' : c_pos  }
#domain_values = {'inner': 3.7,  'outer': 1}
#help(c_disc)
#Draw(c_disc, mesh, 'cdisk')
#domain_values_2 = { 'B': 1.0,  'IF-inner': 1.0,  'IF-outer': 1.0, 'void' : 1.0,  'omega-outer' : 1.0 }

#c_squared = mesh.MaterialCF(domain_values_2)


def CheckSpatial(mesh,lami, wp_mode_space): 

    c_disc = mesh.MaterialCF(domain_values)
    fes = H1(mesh, order=2, dirichlet="R0|R2")
    #fes = H1(mesh, order=1, dirichlet="R0|R2")

    # define trial- and test-functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # the right hand side
    f = LinearForm(fes)
    f +=  lami.real *  wp_mode_space  * v * dx(bonus_intorder =  bonus_intorder) 

    # the bilinear-form
    a = BilinearForm(fes, symmetric=True)
    a +=  c_disc * grad(u)*grad(v) * dx(bonus_intorder =  bonus_intorder) 

    a.Assemble()
    f.Assemble()

    # the solution field
    gfu = GridFunction(fes)
    gfu.Set(wp_mode_space, BND)
    res = f.vec - a.mat * gfu.vec
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * res
    Draw(gfu,mesh,'gfu')
    Draw(wp_mode_space,mesh,'mode')
    Draw(sqrt( (gfu- wp_mode_space)*(gfu-wp_mode_space)), mesh, 'err') 
    l2_abserr =  sqrt (Integrate ( (gfu- wp_mode_space)*(gfu-wp_mode_space), mesh,  order =  2*max(order,4) +  bonus_intorder ))
    l2_norm = sqrt (Integrate (  wp_mode_space*wp_mode_space , mesh, order =  2*max(order,4) +  bonus_intorder  ))
    
    #l2_abserr =  sqrt (Integrate ( (gfu- wp_mode_space)*(gfu-wp_mode_space), mesh, definedon=mesh.Materials("B") ,   order =  2*max(order,4) +  bonus_intorder ))
    #l2_norm = sqrt (Integrate (  wp_mode_space*wp_mode_space , mesh, definedon=mesh.Materials("B"),  order =  2*max(order,4) +  bonus_intorder  ))


    print ("relative L2-error:",  l2_abserr /  l2_norm   )
    return l2_abserr /  l2_norm 
    input("weiter")





def SolveProblem(mesh, N,  order_global, lami, wp_mode_space, omega_str, export_vtk=False,  vtk_str = "" ):
    print("N = ", N)

    c_squared = mesh.MaterialCF(domain_values)
   
    data_stab = 1e4
    #if well_posed:
    #    data_stab = 1e-9 


    stabs = {"data": data_stab,
         "dual": 1,
         "primal": 1e-4,
         "primal-jump":1e1,
         "primal-jump-displ-grad":1e1,
         "Tikh": 1e-18
        }

    q,k,qstar,kstar = get_order(order_global)
    time_order = 2*max(q,qstar)
    print("lami = ", lami)

    #N = 48
    #N = 32
    tstart = 0.0
    tend = 1.0
    #tend = 1.5
    delta_t = tend / N
    # Level-set functions specifying the geoemtry
    told = Parameter(tstart)
    t = told + delta_t * tref

    # define exact solution
    #qpi = pi/4 
    t_slice = [ tstart  + n*delta_t + delta_t*tref for n in range(N)]
    u_exact_slice = [ cos(  sqrt(lami) * t_slice[n] ) * wp_mode_space  for n in range(N)]
    ut_exact_slice = [ sqrt(lami) * (-1) * sin(  sqrt(lami) * t_slice[n] ) * wp_mode_space for n in range(N)]
    
    u_exact_final_time = cos(  sqrt(lami) * tend ) * wp_mode_space 


    qpi = pi/2 
    m_sol = 6 
    #u_exact_slice = [  5*cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]
    #ut_exact_slice = [ -5 * sqrt(2) * m_sol * qpi * sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]
    

    st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=tend,delta_t=delta_t,mesh=mesh,stabs=stabs,
                    t_slice=t_slice, u_exact_slice=u_exact_slice, ut_exact_slice=ut_exact_slice, tstart=tstart, 
                    told=told, well_posed = well_posed, c_squared = c_squared, omega_str=omega_str) 

    #st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=tend,delta_t=delta_t,mesh=mesh,stabs=stabs,
    #                t_slice=t_slice, u_exact_slice=u_exact_slice, ut_exact_slice=ut_exact_slice, tstart=tstart, 
    #                told=told, well_posed = well_posed, c_squared = 1.0)

    st.SetupSpaceTimeFEs()
    st.SetupRightHandSide()
    st.PreparePrecondGMRes()
    
    A_linop = SpaceTimeMat(ndof = st.X.ndof,mult = st.ApplySpaceTimeMat)
    PreTM = SpaceTimeMat(ndof = st.X.ndof,mult = st.TimeMarching)

    st.gfuX.vec.data = GMRes(A_linop, st.f.vec, pre=PreTM, maxsteps = 10000, tol = tol,
                  callback=None, restart=None, startiteration=0, printrates=True)

    #l2_errors_Q = st.MeasureErrors(st.gfuX, domain_B="IF-inner|B")
    l2_errors_Q = st.MeasureErrors(st.gfuX)
    
    diff = GridFunction(st.V_space)
    u_slab_node = GridFunction(st.V_space)
    u_gfu_node = GridFunction(st.V_space)
    #told.Set(st.tend)
    u_slab_node.Set(fix_tref(st.u_exact_slice[st.N-1],1.0))
    #u_slab_node.Set(fix_tref(st.u_exact_slice[st.N-1],st.tend))
    u_gfu_node.vec.FV().NumPy()[:]  =  st.gfuX.components[0].components[st.N-1].vec.FV().NumPy()[ st.q*st.V_space.ndof : (st.q+1)*st.V_space.ndof] 
    diff.vec.FV().NumPy()[:] = np.abs( st.gfuX.components[0].components[st.N-1].vec.FV().NumPy()[ st.q*st.V_space.ndof : (st.q+1)*st.V_space.ndof] 
    -  u_slab_node.vec.FV().NumPy() ) 
    Draw(diff, mesh, 'diff') 
    Draw( u_slab_node, mesh, 'exact') 
    Draw( u_gfu_node , mesh, 'gfu') 
    #input("")


    print("Errors in Q (all) = ", l2_errors_Q)

    if export_vtk:
        VTKOutput(ma=mesh, coefs=[ u_exact_final_time,  u_gfu_node, diff   ],
                      names=["u","gfu","diff"],
                      filename=vtk_str, subdivision=2).Do()


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
 




meshes = [ ] 

Ns = [12,24,48]
#Ns = [12,24]
if n_mode == 1: 
    #Ns = [10,20,40]
    Ns = [10,20,40]
max_nref = 2
#n_refs = 2
for n_refs in range(max_nref+1):
    print("order = ", order)
    #mesh = CreateAnnulusMesh3(maxh=maxh, order_geom = order, extra_refinement=False) 
    mesh = CreateAnnulusMesh3(maxh=maxh, order_geom = order, extra_refinement=False) 
    for i in range(n_refs):
        mesh.Refine()
    #print("order_geom = ", order_geom)
    mesh.Curve(order_geom)
    meshes.append(mesh)


if check_spatial_conv: 
    l2_errors = [ ]
    for mesh in meshes:
        l2_errors.append( CheckSpatial(mesh,lami.real, wp_mode_space) )
    eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
    print("eoc = ", eoc ) 
    input("")

#input("Weiter")
    
for omega_str in ["IF-inner", "omega-outer"]: 
#for omega_str in ["IF-inner"]: 
#for omega_str in ["IF-inner"]: 
#for omega_str in ["omega-outer"]: 
    for order_global in [order]:

        errors = { "delta-t": [],  
                  "Q_all" : [ ] 
                  } 

        for mesh,N in zip(meshes, Ns):
            vtk_str = "" 
            export_vtk = False
            if N == Ns[-1]: 
                export_vtk = True
                vtk_str =  "whispering-gallery-mode-nr" + "{0}".format(n_mode) + "-N{0}".format(N) + "-" + omega_str + "-" + problem_type
            result = SolveProblem(mesh, N, order_global, lami.real, wp_mode_space, omega_str, export_vtk,vtk_str)
            
            errors["delta-t"].append(result[0]) 
            errors["Q_all"].append(result[1])

        print("errors = ", errors) 

        name_str = "whispering-gallery-mode-nr" + "{0}".format(n_mode) + "-" + omega_str + "-" + problem_type + ".dat"
        results = [np.array(errors["delta-t"],dtype=float), np.array( errors["Q_all"],dtype=float)  ]
        header_str = "deltat Q-all"
        np.savetxt(fname ="data/{0}".format(name_str),
                   X = np.transpose(results),
                   header = header_str,
                   comments = '')
