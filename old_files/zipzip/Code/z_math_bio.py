import numpy as np
import scipy as sc
import scipy.sparse as sps
import scipy.sparse.linalg as sps_l
import scipy.optimize as sopt
from functions_bio import *
import matplotlib.pyplot as plt

# The output of optimizer: 
# x: A NumPy array containing the solution of the optimization (i.e., the values of the variables that minimize the objective function).
# success: A boolean indicating whether the optimizer exited successfully.
# status: An integer representing the termination status of the optimizer. The meaning of the value depends on the underlying solver.
# message: A string describing the cause of the termination.
# fun: The value of the objective function at the solution.
# jac: The Jacobian (gradient) of the objective function at the solution, if it was computed.
# hess: The Hessian (matrix of second derivatives) of the objective function at the solution, if it was computed.
# nfev: The number of function evaluations.
# nit: The number of iterations performed by the optimizer.

################################################################################
def zcompute_yz(u,params):
################################################################################
    m=params['m']
    n=params['n']
    ntotal=n*m
#
    omega=params['omega'];
    omega_v=params['omega_v'];
    s=params['s']
    Psi_S=params['Psi_S']
    XN=params['mN']
    a=params['ma']
    eps_1=params['eps_all'][0]; eps_2=params['eps_all'][1]; eps_3=params['eps_all'][2]
#
    chi_1=u[0:ntotal]
    chi_2=u[ntotal:2*ntotal]
    chi_3=u[2*ntotal:3*ntotal]
# compute Y_{ij} and Z_{ij}
    D_Psi_S=sps.diags([Psi_S.reshape((ntotal,))],[0],(ntotal,ntotal)) # this is diag(psi_S)
    d_s=sps.diags([s.reshape((n,))],[0],(n,n))
    d_beta=sps.diags([beta.reshape((n,))],[0],(n,n))
    s_a_beta=(d_s @ (a @ d_beta)).reshape(ntotal,1)
    d_s_a_beta=sps.diags([s_a_beta.reshape(ntotal,)],[0],(ntotal,ntotal))
    #print(d_s_a_beta)
    ch1=chi_1.reshape((ntotal,1)); ch2=chi_2.reshape((ntotal,1));ch3=chi_3.reshape((ntotal,1))
    p1=ch1/(omega_v+eps_2*ch2)
    p2=(eps_2/omega)*ch2
    p3=1+(eps_3/omega)*ch3
    SN=(1+p1*(1+p2+p2*p3))
    SN=1./SN
    VN=p1*SN
    PN=p2*VN
    RN=p3*PN
#####    GN=(1.-eps_2)*VN+(1.-eps_3)*PN
    GN=(1.-eps_1)*VN + PN
    ## yz is the Schur product 
    D=(D_Psi_S @ d_s_a_beta)
    #  no need to reshape; work with sparse 
    y = (D @ SN) #.reshape((nloc,mloc))
    z = (D @ GN)#.reshape((nloc,mloc)) 
    return y,z
###############################################################################
def one_eig(X):
    e,v=sps_l.eigs(X, k=1,which='LM',tol=0, return_eigenvectors=True)
    e=np.real(e[0])
    v=v.reshape((np.shape(v)[0],))
    v=np.abs(np.real(v))
    return e,v

# def rho0(u,wk,phi,params):
#     y,z=zcompute_yz(u,params)
#     d1=np.empty(shape=(2*ntotal,),dtype=float)
#     d1[::2]=1./y.reshape((ntotal,)); d1[1::2]=1./(z.reshape((ntotal,)))
# #    print('OK',D.shape,A.shape,n2)
#     wp=(wk*phi)/np.dot(wk,phi)
#     rho1=np.dot(d1,wp)
#     return -rho1
#########################################################################
def rho2(u,phi,Astay,params):
#    u() contains chi1chi2chi3 flattened
    m=params['m'];n=params['n']
    ntotal=m*n
    n2=2*ntotal
    chi1=u[0:ntotal]
    chi2=u[ntotal:2*ntotal]
    chi3=u[2*ntotal:3*ntotal]
    yy,zz=zcompute_yz(u,params)
    ## yy and zz computed; compute d.
    d1=np.empty(shape=(n2,),dtype=float)
    d1[::2]=1./(yy.reshape((ntotal,))); d1[1::2]=1./(zz.reshape((ntotal,)))
    #D1=sps.diags([d1],[0],(n2,n2))
    D=sps.diags([1./d1],[0],(n2,n2))
#    print('OK',D.shape,A.shape,n2)
    rho,wk=one_eig(D@Astay)
    wk=wk/(np.linalg.norm(wk,ord=1))
    # wp=(wk*phi)/np.dot(wk,phi)
    # rho1=np.dot(d1,wp)
#    print(rho,end=",")# ,np.dot(wk,phi))
    print(rho)# ,np.dot(wk,phi))
#    print('d1=',1./d1)
    return rho
#########################################################################
def sol_rho(u,Astay,params):
#    u() contains chi1chi2chi3 flattened
    m=params['m'];n=params['n']
    ntotal=m*n
    n2=2*ntotal
    chi1=u[0:ntotal]
    chi2=u[ntotal:2*ntotal]
    chi3=u[2*ntotal:3*ntotal]
    yy,zz=zcompute_yz(u,params)
    ## yy and zz computed; compute d.
    d1=np.empty(shape=(n2,),dtype=float)
    d1[::2]=1./(yy.reshape((ntotal,))); d1[1::2]=1./(zz.reshape((ntotal,)))
    #D1=sps.diags([d1],[0],(n2,n2))
    D=sps.diags([1./d1],[0],(n2,n2))
#    print('OK',D.shape,A.shape,n2)
    DA=D@Astay; del D;
    rho,wk=one_eig(DA)
    wk=wk/(np.linalg.norm(wk,ord=1))
    return rho,wk,DA
#########################################################################
def constr(params):
    n=params['n'];m=params['m'];ntotal=n*m
    mS=params['mS'];mP=params['mP'];mV=params['mV'];
    nrows=m # this should be 3*m for full constrained optimization 
    lb=np.zeros(nrows,dtype=float)
    lb[0:m]=np.array(params['b1'],dtype=float)
    AA=sps.coo_matrix((m,3*ntotal), dtype=float)
    for i in range(nrows):
        rmax=1./np.max(mS[i,:])
        lb[i]=rmax*lb[i] ## scale the rhs
        jj=np.arange(start=i*n,stop=(i+1)*n,dtype=int)
        ii=i*np.ones((m,),dtype=int)
        AA=AA+sps.coo_matrix((rmax*mS[i,:],(ii,jj)),shape=(m,3*ntotal))
    # checking np.savetxt("AA.dat",AA.toarray())
    ub=lb
    return lb,ub,AA
def call_rho(rh_val):
    global rh
    rh.append(rh_val)
##############################MAIN
np.set_printoptions(precision=16)
np.set_printoptions(linewidth=np.inf)

params=params_bio1()
n=params['n'];m=params['m']; ntotal=m*n
params['eta']=np.concatenate((np.array([0.,0.]).reshape(2,),params['eta']),axis=0)
for key in params.keys():
    comm=key+"""=params['"""+key+"""']"""
    exec(comm,None,None)

# READ input matrices and vectors and set up the stage
data_files={"C":"cMatrix",
            "XN":"nMatrix",
            "a":"aMatrix",
            "chi1":"chi1Matrix",
            "chi2":"chi2Matrix",
            "chi3":"chi3Matrix",
           # "F":"Fmatrix",
            "SN0":"snMatrix",
            "VN0":"vnMatrix",
            "PN0":"pnMatrix",
            "Y0":"yMatrix",
            "Z0":"zMatrix",
            }
dmat=read_data_files(data_files)
for key in dmat.keys():
    comm=key+"""=dmat['"""+key+"""']"""
    exec(comm,None,None)
del dmat
del data_files

data_files={"C":"cMatrix",
            "CCm":"ccmMatrix",
            "CCmat":"ccmatMatrix",
            "beta":"betaVector",
            "Q1":"qMatrix",
            "H":"hMatrix",
            "NGM":"NGM",
            "mS":"s1Matrix",
            "mP":"p1Matrix",
            "mV":"v1Matrix",
            "b1":"b1Vector"
            }
dmat=read_data_files(data_files)
for key in dmat.keys():
    comm=key+"""=dmat['"""+key+"""']"""
    exec(comm,None,None)
for key in dmat.keys():
    comm="""params['"""+key+"""']=dmat['"""+key+"""']"""
    exec(comm,None,None)
del dmat
del data_files
#print(np.linalg.norm(C.reshape((16,16))-CCm))
# print(np.shape(C),np.shape(CCm),np.shape(CCmat))
# exit(2)
lb,ub,AA=constr(params)
Co123=sc.optimize.LinearConstraint(AA,lb,ub) #keep_feasible=True)
bounds = [(0, np.inf) for k in range(3*ntotal)]

u=np.ones(shape=(3*ntotal,),dtype=float)
ax=AA@u
for i in range(m):
    u[i*n:(i+1)*n]=lb[i]/ax[i]
print('''%%Init: violation(constr) = ''',np.linalg.norm(AA@u-lb))
########### guess set; delete from params. 
#del params['chi1'];del params['chi2'];del params['chi3']
# for i in range(3):
#     print('''chi'''+str(i+1)+'''=''',
#                u[i*ntotal:(i+1)*ntotal].reshape(n,m))

# YY,ZZ=zcompute_yz(u,params)
# print('YY=',YY.reshape(n,m))
# print('ZZ=',ZZ.reshape(n,m))
# exit(22)

n2=2*ntotal
yy,zz=zcompute_yz(u,params)
d1=np.empty(shape=(n2,),dtype=float)
d1[::2]=1./(yy.reshape((ntotal,))); d1[1::2]=1./(zz.reshape((ntotal,)))
D=sps.diags([d1],[0],(2*ntotal,2*ntotal)) ## Omega in the write-up
D1=sps.diags([1./d1],[0],(2*ntotal,2*ntotal)) ## 
Astay=D @ sps.coo_matrix(NGM)
# print(Astay.toarray())
rhoA,phiA=one_eig(Astay.T)


opts = {'maxiter': 1300,'ftol':2.**(-40)}
print('rho=[',end="")# ,np.dot(wk,phi))
op = sopt.minimize(rho2, u, method='SLSQP', options=opts,constraints=Co123,bounds=bounds,
                       args=(phiA,Astay,params))
##########################

rho_end,v,NGM_opt=sol_rho(op.x,Astay,params)

print(rho_end,'];')# ,np.dot(wk,phi))

print('%%Success           = ',op.success)# A boolean indicating whether the optimizer exited successfully.
print('%%Status            = ', op.status)# An integer representing the termination status of the optimizer. The meaning of the value depends on the underlying solver.
print('%%msg               = ',op.message)# A string describing the cause of the termination.
print('%%rho(end)          = ',op.fun)# The value of the objective function at the solution.
#print(op.jac)# The Jacobian (gradient) of the objective function at the solution, if it was computed.
#####print(op.hess)# The Hessian (matrix of second derivatives) of the objective function at the solution, if it was computed.
print('%%num(fun_eval)     = ',op.nfev)# The number of function evaluations.
print('%%num(iter)         = ',op.nit)# The number of iterations performed by the optimizer.
print('%%violation(constr) = ',np.linalg.norm(AA@op.x-lb))
#
################################################
#print("""chi=""",op.x,';')
print("\nrho = ",rho_end)
y,z=zcompute_yz(op.x,params)
#print("y=",y.reshape(n*m,))
#print("z=",z.reshape(n*m,))
np.savetxt('''eigenvector'''+'''.dat''',v)
np.savetxt("NGM_opt.dat",NGM_opt.toarray())
#np.savetxt('''Y'''+'''.dat''',
#           y.reshape(m,n));
#np.savetxt('''Z'''+'''.dat''',
#           z.reshape(m,n));
for i in range(3):
    np.savetxt('''chi'''+str(i+1)+'''.dat''',
               op.x[i*ntotal:(i+1)*ntotal].reshape(n,m))
np.savetxt("Astay.dat",Astay.toarray())

#print(rh)