import numpy as np
import scipy.sparse as sps
def power_method(A=np.random.rand(3,3),tol=1e-8,iprt=False):
    ##
    ## [yk,lambdak]=power_method(A,tol)
    ## power method for eigenvalues (tol=toleraance)
    ## it does not work for complex eigenvalues. 
    maxiter=1000
    n=np.shape(A)[0]
    if(not sps.issparse(A)):
        A=sps.csr_matrix(A,shape=(n,n))
    xk=np.random.randn(n,1)
    iter=0;
    # ## if inverse power method: use the LU decomposition
    # ## with partial pivoting
    # if(flag):
    #     L,U,P=palu(A);
    # end
    if(iprt):
        while(1):
            yk=(1./np.linalg.norm(xk))*xk
            xk=A @ yk
            ek=yk.T @ xk ##
            ## check convergence
            stop0=np.linalg.norm(xk-ek*yk); ##
            print('iter=',iter,' ek=',ek,' |A*y-lambda * y|=', stop0)
            iter=iter+1
            if(stop0<tol):
                flag=0
                break
            if(iter>maxiter): 
                flag=1
                break
    else:
            while(1):
            yk=(1./np.linalg.norm(xk))*xk
            xk=A @ yk
            ek=yk.T @ xk ##
            ## check convergence
            stop0=np.linalg.norm(xk-ek*yk); ##
            print('iter=',iter,' ek=',ek,' |A*y-lambda * y|=', stop0)
            iter=iter+1
            if(stop0<tol):
                flag=0
                break
            if(iter>maxiter):
                flag=1 
                break
    return np.squeeze(ek),np.squeeze(xk),flag
