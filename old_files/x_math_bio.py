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
np.set_printoptions(precision=16)
np.set_printoptions(linewidth=np.inf)

params=params_bio1()
n=params['n'];m=params['m']; ntotal=m*n
params['eta']=np.concatenate((np.array([0.,0.]).reshape(2,),params['eta']),axis=0)
for key in params.keys():
    comm=key+"""=params['"""+key+"""']"""
    exec(comm,None,None)
# READ input matrices and vectors and set up the stage
data_files={"Ac":"Ac",
            "DYZ":"DYZ",
            "Y1":"Y1",
            "Z1":"Z1",
            "NGM":"NGM"
            }
dmat=read_data_files(data_files,dir0='''new_nb_Data/''',ext='''.csv''')
for key in dmat.keys():
    comm=key+"""=dmat['"""+key+"""']"""
    exec(comm,None,None)
for key in dmat.keys():
    comm="""params['"""+key+"""']=dmat['"""+key+"""']"""
    exec(comm,None,None)
del dmat
del data_files
#n2=size(Ac,1);m2=size(Ac,2);
E=DYZ@Ac-NGM
en=np.linalg.norm(E,ord=1)
print('en=',en,flush=True)
l1_E = np.abs(E).sum(axis=0).max()
n2=Ac.shape[0]
m2=Ac.shape[1]
n_p=n2//2 
m_p=m2//2
I = np.identity(n2)
#print(n_p,m_p,n2,m2)
#chk1=np.linalg.norm(,1)
## allocate u

# bounds=[(0., np.inf) for k in range(3*ntotal)]
# bounds[0:ntotal] = [(0.,np.inf) for k in range(ntotal)]
# Ac,_=get_constA(A_in,params)
# u=np.ones(shape=(3*ntotal,),dtype=float)
# ############### equality constraints
# lb,AA=constr(params)

# #initial guess
# ax=AA@u ## this is the row sum of AA
# for i in range(3*m):
#     u[i*n:(i+1)*n]=lb[i]/ax[i]
# u000=u.copy()
# print('''%%Init: violation(constr) = ''',np.linalg.norm(AA@u-lb))
# #
# n2=2*ntotal
# rhoA,phiA=one_eig(Ac.T)
# print(rhoA)
# print(phiA)
# exit(1)
# opts = {'maxiter': 1300,'ftol':2.**(-20)}
# print('rho=[',end="")# ,np.dot(wk,phi))

# cons = ({'type': 'eq', 'fun': lambda x: AA @ x - lb})

# op = sopt.minimize(rho2, u, method='SLSQP', options=opts,constraints=cons,bounds=bounds,
#                        args=(phiA,Aconst,params))
# ##########################

# rho_end,v,NGM_opt=sol_rho(op.x,Aconst,params)

# print(rho_end,'];')# ,np.dot(wk,phi))

# print('%%Success           = ',op.success)# A boolean indicating whether the optimizer exited successfully.
# print('%%Status            = ', op.status)# An integer representing the termination status of the optimizer. The meaning of the value depends on the underlying solver.
# print('%%msg               = ',op.message)# A string describing the cause of the termination.
# print('%%rho(end)          = ',op.fun)# The value of the objective function at the solution.
# #print(op.jac)# The Jacobian (gradient) of the objective function at the solution, if it was computed.
# #####print(op.hess)# The Hessian (matrix of second derivatives) of the objective function at the solution, if it was computed.
# print('%%num(fun_eval)     = ',op.nfev)# The number of function evaluations.
# print('%%num(iter)         = ',op.nit)# The number of iterations performed by the optimizer.
# print('%%violation(constr) = ',np.linalg.norm(AA@op.x-lb))
# #
# ################################################
# #print("""chi=""",op.x,';')
# print('||uopt-u0||',np.linalg.norm(op.x-u000))
# print("\nrho = ",rho_end)
# #------------------------------------------------------------------------------------------------
# #y,z=acompute_yz(op.x,params)
# #print("y=",y.reshape(n*m,))
# #print("z=",z.reshape(n*m,))
# #np.savetxt('''eigenvector'''+'''.dat''',v)
# #np.savetxt("NGM_opt.dat",NGM_opt.toarray())
# #np.savetxt('''Y'''+'''.dat''',
# #           y.reshape(m,n));
# #np.savetxt('''Z'''+'''.dat''',
# #           z.reshape(m,n));
# for i in range(3):
#     np.savetxt('''chi'''+str(i+1)+'''.dat''',
#                op.x[i*ntotal:(i+1)*ntotal].reshape(n,m))
# #np.savetxt("Aconst.dat",sps.coo_matrix(Aconst).toarray())


# #print(rh)