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
def get_constA(A_in,params):
################################################################################
# computes the constant matrix that is a factor in the optimization. 
# writing NGM=D(chi)@(D_1@A) where D_1@A is computed here. 
    m=params['m']
    n=params['n']
    ntotal=n*m
    omega=params['omega'];
    omega_v=params['omega_v'];
    s=params['s']
    Psi_S=params['Psi_S']
    XN=params['mN']
    a=params['ma']
    eps_1=params['eps_all'][0]; eps_2=params['eps_all'][1]; eps_3=params['eps_all'][2]
    D_Psi_S=sps.diags([Psi_S.reshape((ntotal,))],[0],(ntotal,ntotal)) # this is diag(psi_S)
    d_s=sps.diags([s.reshape((n,))],[0],(n,n)) #this is diag(s)
    d_beta=sps.diags([beta.reshape((n,))],[0],(n,n))
    s_a_beta=(d_s @ (a @ d_beta)).reshape(ntotal,1)
    d_s_a_beta=sps.diags([s_a_beta.reshape(ntotal,)],[0],(ntotal,ntotal))
    #print(d_s_a_beta)
    n2=2*ntotal
    dc=(D_Psi_S @ d_s_a_beta).diagonal(k=0)
    d=np.zeros(n2,dtype=float)
    d[::2]=dc.reshape((ntotal,)); 
    d[1::2]=dc.reshape((ntotal,));
    D=sps.diags([d],[0],(n2,n2))
    return  D@A_in,d
####################################################
def acompute_yz(u,params):
    # compute y and z but after the diagonal scaling by a() and psiS is removed
################################################################################
    m=params['m']
    n=params['n']
    ntotal=n*m
#
    omega=params['omega'];
    omega_v=params['omega_v'];
    s=params['s']
    # Psi_S=params['Psi_S']
    # XN=params['mN']
    # a=params['ma']
    eps_1=params['eps_all'][0]; eps_2=params['eps_all'][1]; eps_3=params['eps_all'][2]
#
    ch1=u[0:ntotal].reshape((ntotal,));
    ch2=u[ntotal:2*ntotal].reshape((ntotal,));
    ch3=u[2*ntotal:3*ntotal].reshape((ntotal,))
#
    p1=ch1/(omega_v+eps_2*ch2)
    p2=(eps_2/omega)*ch2
    p3=1+(eps_3/omega)*ch3
    SN=(1+p1*(1+p2+p2*p3))
    SN=1./SN
    VN=p1*SN
    PN=p2*VN
    RN=p3*PN
    GN=(1.-eps_1)*VN + PN
    if(np.min(GN)<1e-14):    
        print('||GN||=',np.min(ch1),np.min(VN))
    y = SN
    z = GN
    #print(np.linalg.norm((1-eps_1)*(1-SN)-GN,ord=1))
    return y,z
###############################################################################
def one_eig(X):
    e,v=sps_l.eigs(X, k=5,which='LM',tol=0, return_eigenvectors=True)
    p=np.argsort(np.abs(e))[::-1]
    e=e[p]; v=np.real(v[:,p[0]])
    #print(np.real(e))
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
def rho2(u,phi,Ac,params):
#    u() contains chi1chi2chi3 flattened
    m=params['m'];n=params['n']
    ntotal=m*n
    n2=2*ntotal
    chi1=u[0:ntotal]
    chi2=u[ntotal:2*ntotal]
    chi3=u[2*ntotal:3*ntotal]
    yy,zz=acompute_yz(u,params)
    ## yy and zz computed; compute d.
    d1=np.empty(shape=(n2,),dtype=float)
    d1[::2]=(yy.reshape((ntotal,))); d1[1::2]=(zz.reshape((ntotal,)))
    D=sps.diags([d1],[0],(n2,n2))
    rho,_=one_eig(D@Ac)
    print(rho)
#    print("\nyz=",yy/zz)
    return rho
#########################################################################
def sol_rho(u,Ac,params):
#    u() contains chi1chi2chi3 flattened
    m=params['m'];n=params['n']
    ntotal=m*n
    n2=2*ntotal
    chi1=u[0:ntotal]
    chi2=u[ntotal:2*ntotal]
    chi3=u[2*ntotal:3*ntotal]
    yy,zz=acompute_yz(u,params)
    ## yy and zz computed; compute d.
    d1=np.empty(shape=(n2,),dtype=float)
    d1[::2]=(yy.reshape((ntotal,))); 
    d1[1::2]=(zz.reshape((ntotal,)))
    D=sps.diags([d1],[0],(n2,n2))
    DA=sps.coo_matrix(D@Ac); del D;
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
    # ub=lb
    # return lb,ub,AA
    return lb,AA
#------------------------------------------------------------------------------------------------
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
            "A_in":"Ac_r"
            }
            # "zchi1":"zchi1Matrix",
            # "zchi2":"zchi2Matrix",
            # "zchi3":"zchi3Matrix",
            # "F":"Fmatrix",
            # "SN0":"snMatrix",
            # "VN0":"vnMatrix",
            # "PN0":"pnMatrix",
            # "Y0":"zY",
            # "Z0":"zZ",
            # "Y1":"zY1",
            # "Z1":"zZ1",
            # "om":"zoMatrix",
            # "om1":"zo1Matrix",
            # }
dmat=read_data_files(data_files)
for key in dmat.keys():
    comm=key+"""=dmat['"""+key+"""']"""
    exec(comm,None,None)
del dmat
del data_files

data_files={"C":"cMatrix",
            "beta":"betaVector",
            "mS":"s1Matrix_r",
            "mP":"p1Matrix_r",
            "mV":"v1Matrix_r",
            "b1":"b1Vector"
            # "CCm":"ccmMatrix",
            # "CCmat":"ccmatMatrix",
            # "Q1":"qMatrix",
            # "H":"hMatrix",
            # "zNGM":"zNGM",
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
## allocate u
#############################################################################
# u=np.empty(3*ntotal,)
# this below is used ONLY if zchi1-3 are saved. look at eigenvalues_H_ZF5_v2.nb 
#for i in range(m):
#    u[0:ntotal]=zchi1.reshape(ntotal,)
#    u[ntotal:2*ntotal]=zchi2.reshape(ntotal,)
#    u[2*ntotal:3*ntotal]=zchi3.reshape(ntotal,)
# Ac,_=get_constA(A_in,params)
# y,z=acompute_yz(u,params)
# d1=np.zeros(2*ntotal,dtype=float)
# d1[::2]=y.reshape((ntotal,)); 
# d1[1::2]=z.reshape((ntotal,));
# Dimp=sps.diags([d1],[0],(2*ntotal,2*ntotal))

# print(np.linalg.norm(Dimp@(Ac)-NGM,ord=1)) ##checked
# exit(1)
#############################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#print(np.linalg.norm(C.reshape((16,16))-CCm))
# print(np.shape(C),np.shape(CCm),np.shape(CCmat))
# exit(2)
#Co123=sc.optimize.LinearConstraint(AA,lb,ub) #keep_feasible=True)
bounds=[(0., np.inf) for k in range(3*ntotal)]
bounds[0:ntotal] = [(0.,np.inf) for k in range(ntotal)]
Ac,_=get_constA(A_in,params)
u=np.ones(shape=(3*ntotal,),dtype=float)
############### equality constraints
lb,AA=constr(params)

#initial guess
ax=AA@u ## this is the row sum of AA
for i in range(3*m):
    u[i*n:(i+1)*n]=lb[i]/ax[i]
u000=u.copy()
print('''%%Init: violation(constr) = ''',np.linalg.norm(AA@u-lb))
#
n2=2*ntotal
rhoA,phiA=one_eig(Ac.T)
print(rhoA)
print(phiA)
exit(1)
opts = {'maxiter': 1300,'ftol':2.**(-20)}
print('rho=[',end="")# ,np.dot(wk,phi))

cons = ({'type': 'eq', 'fun': lambda x: AA @ x - lb})

op = sopt.minimize(rho2, u, method='SLSQP', options=opts,constraints=cons,bounds=bounds,
                       args=(phiA,Aconst,params))
##########################

rho_end,v,NGM_opt=sol_rho(op.x,Aconst,params)

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
#------------------------------------------------------------------------------------------------
#y,z=acompute_yz(op.x,params)
#print("y=",y.reshape(n*m,))
#print("z=",z.reshape(n*m,))
#np.savetxt('''eigenvector'''+'''.dat''',v)
#np.savetxt("NGM_opt.dat",NGM_opt.toarray())
#np.savetxt('''Y'''+'''.dat''',
#           y.reshape(m,n));
#np.savetxt('''Z'''+'''.dat''',
#           z.reshape(m,n));
for i in range(3):
    np.savetxt('''chi'''+str(i+1)+'''.dat''',
               op.x[i*ntotal:(i+1)*ntotal].reshape(n,m))
#np.savetxt("Aconst.dat",sps.coo_matrix(Aconst).toarray())


#print(rh)