'''
Forming the matrix for the mathbio problem.

Authors: blah @ nsf.gov
'''

import numpy as np
import scipy.sparse as sps

### all parameters are in params_bio.py so we can import all of them:
from functions_bio import *
def old_rho0(z):
    # z[0:2] is chi1,chi2,chi3
    # phi is z[3] (eigenvector of A)
    # the rest are computed
    n=int(np.floor(z[-1]+0.1))
    n2=2*n
    print('z=',z[-1])
    print(z[0:3*n])
    chi=(z[0:3*n].reshape((3,n))).T
    # print(chi)
    phi=z[3*n:3*n+n2]
    # exit(1)
    yy=chi[:,0]/(1+chi[:,1])+1e-1
    zz=yy*(chi[:,2]/(1+chi[:,0])+1.)
    ## yy and zz computed; compute d.
    d1=np.empty(shape=(n2,),dtype=float)
    d1[::2]=1./yy; d1[1::2]=1./zz
    #D1=sps.diags([d1],[0],(n2,n2))
    D=sps.diags([1./d1],[0],(n2,n2))
#    print('OK',D.shape,A.shape,n2)
    rho,x=one_eig(D@A)
    x=x/(np.linalg.norm(x,ord=1))
    x=x/(np.linalg.norm(x,ord=1))
    xp=(x*phi)/np.dot(x,phi)
    rho1=np.dot(d1,xp)
    print('rho=',rho,rho1)
    print('d1=',1./d1)
    return -rho1
#########################################################################
params=params_bio0()
## add two zeros to eta
params['eta']=np.concatenate((np.array([0.,0.]).reshape(2,),params['eta']),axis=0)
for key in params.keys():
    comm=key+"""=params['"""+key+"""']"""
    exec(comm,None,None)

# READ input matrices and vectors and set up the stage
data_files={"C":"cMatrix",
            "XN":"nMatrix",
            "beta":"betaVector",
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
            "Q1":"qMatrix",
            "H":"hMatrix",
            "NGM":"NGM"
            }
dmat=read_data_files(data_files)
for key in dmat.keys():
    comm=key+"""=dmat['"""+key+"""']"""
    exec(comm,None,None)
del dmat
del data_files

# print(np.linalg.norm(C.reshape((16,16))-CCm))
# print(np.shape(C),np.shape(CCm),np.shape(CCmat))
# exit(2)

NGM00=np.empty(shape=(32,32))
ix=0;iy=ix+8
for i in range(4):
    HT=(H[i,:].transpose((1,0,2)))
    HT=np.reshape(HT,(256,))   
    HT=np.reshape(HT,(8,32))
    NGM00[ix:iy,:]=HT
    ix=ix+np.shape(HT)[0];iy=ix+8
    print(ix,iy)
    del HT
################ end input
##
#chi1=np.random.rand(4,4);chi2=np.random.rand(4,4);chi3=np.random.rand(4,4)
##
#y,z=compute_yz(chi1,chi2,chi3,a,beta,XN,params)
#print("ey=",y-Y0,"\nez=",Z0-z)

# (i,p,j,q)-->(i,j,p,q)
C=C.transpose((0,2,1,3))
ndimC=np.shape(C) 
n=ndimC[0]
m=ndimC[1]
ntotal=n*m
C=C.reshape(ntotal,ntotal)
#
print('Shape(NGM00)=',np.shape(NGM00),'; err=',np.linalg.norm(NGM00-NGM))
np.savetxt("NGM00.dat",NGM00)
np.savetxt("Y.dat",Y0.reshape(16,1))
np.savetxt("Z.dat",Z0.reshape(16,1))
exit(1)
F_chk=F.transpose((0,2,1,3,4,5)).copy()
del F # F was read, so here we delete it as it is not needed. 
ndimF=np.shape(F_chk);
neta=np.shape(eta)[0]
nF=ntotal*neta
F_chk=sps.csr_matrix(F_chk.reshape(nF,nF))
np.savetxt("F_chk.dat",F_chk.toarray())
F_new=compute_F(y,z,C,eta,params)
#print(np.shape(F1))
np.savetxt("F_new.dat",F_new.toarray())
############################################################
