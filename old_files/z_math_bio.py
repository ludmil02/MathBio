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
    chi_1=u[0:ntotal];chi_2=u[ntotal:2*ntotal];chi_3=u[2*ntotal:3*ntotal]
# compute Y_{ij} and Z_{ij}
    D_Psi_S=sps.diags([Psi_S.reshape((ntotal,))],[0],(ntotal,ntotal)) # this is diag(psi_S)
    d_s=sps.diags([s.reshape((n,))],[0],(n,n)) #this is diag(s)
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
    if(np.min(GN)<1e-14):    
        print('||GN||=',np.min(ch1),np.min(VN))
        #print(ch1)
    #    exit(1)
    ## yz is the Schur product 
    D=(D_Psi_S @ d_s_a_beta)
    #  no need to reshape; work with sparse 
    y = (D @ SN) #.reshape((nloc,mloc))
    z = (D @ GN)#.reshape((nloc,mloc)) 
    return y,z
################################################################################
def one_eig(X):
    e,v=sps_l.eigs(X, k=5,which='LM',tol=0, return_eigenvectors=True)
    p=np.argsort(np.abs(e))[::-1]
    e=e[p]; v=np.real(v[:,p[0]])
    print(np.real(e))
    e=np.real(e[0])
    #print(e,v,np.shape(v))
    v=v.reshape((np.shape(v)[0],))
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
    #print(np.min(yy.reshape(ntotal,)),np.min(zz.reshape(ntotal,)))
    d1[::2]=(yy.reshape((ntotal,))); d1[1::2]=(zz.reshape((ntotal,)))
    #D1=sps.diags([d1],[0],(n2,n2))
    D=sps.diags([d1],[0],(n2,n2))
#    print('OK',D.shape,A.shape,n2)
    rho,_=one_eig(D@Astay)
    # rho,wk=one_eig(D@Astay)
    # wk=wk/(np.linalg.norm(wk,ord=1))
    # wp=(wk*phi)/np.dot(wk,phi)
    # rho1=np.dot(d1,wp)
#    print(rho,end=",")# ,np.dot(wk,phi))
    print(rho)# ,np.dot(wk,phi))
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
    d1[::2]=(yy.reshape((ntotal,))); d1[1::2]=(zz.reshape((ntotal,)))
    #D1=sps.diags([d1],[0],(n2,n2))
    D=sps.diags([d1],[0],(n2,n2))
#    print('OK',D.shape,A.shape,n2)
    DA=D@Astay; del D;
    rho,wk=one_eig(DA)
    wk=wk/(np.linalg.norm(wk,ord=1))
    return rho,wk,DA
#########################################################################
def constr(params):
    n=params['n'];m=params['m'];ntotal=n*m
    mS=params['mS'];mP=params['mP'];mV=params['mV'];
    np.random.seed(61)
    mP_random=np.random.rand(np.shape(mP)[0],np.shape(mP)[1])
    np.random.seed(61)
    mV_random=np.random.rand(np.shape(mP)[0],np.shape(mP)[1])
    nrows=3*m # this should be 3*m for full constrained optimization 
    lb=np.zeros(nrows,dtype=float)
    lb[0:m]=np.array(params['b1'],dtype=float)
    AA=sps.coo_matrix((3*m,3*ntotal), dtype=float)
    for i in range(m):
        rmax=1./np.max(mS[i,:])
        lb[i]=rmax*lb[i] ## scale the rhs
        jj=np.arange(start=i*n,stop=(i+1)*n,dtype=int)
        ii=i*np.ones((m,),dtype=int)
        AA=AA+sps.coo_matrix((rmax*mS[i,:],(ii,jj)),shape=(3*m,3*ntotal))
    # checking np.savetxt("AA.dat",AA.toarray())
    for i in range(m,2*m):
        rmax=1./np.max(mP_random[i-m,:])
        lb[i]=rmax*lb[i] ## scale the rhs
        jj=np.arange(start=i*n,stop=(i+1)*n,dtype=int)
        ii=i*np.ones((m,),dtype=int)
        AA=AA+sps.coo_matrix((rmax*mP_random[i-m,:],(ii,jj)),shape=(3*m,3*ntotal))
    for i in range(2*m,3*m):
        rmax=1./np.max(mV_random[i-2*m,:])
        lb[i]=rmax*lb[i] ## scale the rhs
        jj=np.arange(start=i*n,stop=(i+1)*n,dtype=int)
        ii=i*np.ones((m,),dtype=int)
        AA=AA+sps.coo_matrix((rmax*mV_random[i-2*m,:],(ii,jj)),shape=(3*m,3*ntotal))
    #checking 
#    np.savetxt("AA.dat",AA.toarray())
    ub=lb
    #print(ub)
    #exit(10)
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
           # "SN0":"snMatrix",
           # "VN0":"vnMatrix",
           # "PN0":"pnMatrix",
           # "Y0":"yMatrix",
           # "Z0":"zMatrix",
            }
dmat=read_data_files(data_files)
for key in dmat.keys():
    comm=key+"""=dmat['"""+key+"""']"""
    exec(comm,None,None)
del dmat
del data_files

data_files={"C":"cMatrix",
           # "CCm":"ccmMatrix",
           # "CCmat":"ccmatMatrix",
            "beta":"betaVector",
            # "Q1":"qMatrix",
            # "H":"hMatrix",
            "NGM":"NGMx",
            "As1":"Astayz",
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
#Co123=sc.optimize.LinearConstraint(AA,lb,ub) #keep_feasible=True)
bounds=[(0., np.inf) for k in range(3*ntotal)]
bounds[0:ntotal] = [(0.,np.inf) for k in range(ntotal)]
u=np.ones(shape=(3*ntotal,),dtype=float)
ax=AA@u
for i in range(3*m):
    u[i*n:(i+1)*n]=lb[i]/ax[i]
for i in range(m):
    u[0:ntotal]=chi1.reshape(ntotal,)
    u[ntotal:2*ntotal]=chi2.reshape(ntotal,)
    u[2*ntotal:3*ntotal]=chi3.reshape(ntotal,)
u000=u.copy()
print('''%%Init: violation(constr) = ''',np.linalg.norm(AA@u-lb))
########### guess set; delete from params. 
# del chi1; del chi2 ; del chi3
#for i in range(3):
#     print('''chi'''+str(i+1)+'''=''',
#                u[i*ntotal:(i+1)*ntotal].reshape(n,m))

#YY,ZZ=zcompute_yz(u,params)
#print('YY=',YY.reshape(n,m))
#print('ZZ=',ZZ.reshape(n,m))
#exit(22)

n2=2*ntotal
yy,zz=zcompute_yz(u,params)
print(1./yy[0],zz[0],flush=True)
exit(10)
d1=np.empty(shape=(n2,),dtype=float)
d1[::2]=1./(yy.reshape((ntotal,))); d1[1::2]=1./(zz.reshape((ntotal,)))
D=sps.diags([d1],[0],(2*ntotal,2*ntotal)) ## Omega in the write-up
D1=sps.diags([1./d1],[0],(2*ntotal,2*ntotal)) ## 
Astay=D @ sps.coo_matrix(NGM)
Astay=sps.coo_matrix(As1.copy())
print('norm(Astay-As1)=',np.linalg.norm(Astay.toarray()-As1,ord=1))
# np.savetxt("Astay.dat",Astay.toarray())
# np.savetxt("As1.dat",As1)
# print(Astay.toarray())
rhoA,phiA=one_eig(Astay.T)

opts = {'maxiter': 1300,'ftol':2.**(-20)}
print('rho=[',end="")# ,np.dot(wk,phi))

cons = ({'type': 'eq', 'fun': lambda x: AA @ x - lb})

op = sopt.minimize(rho2, u, method='SLSQP', options=opts,constraints=cons,bounds=bounds,
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
print('||uopt-u0||',np.linalg.norm(op.x-u000))
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