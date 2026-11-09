'''
Forming the matrix for the mathbio problem.

Authors: blah @ nsf.gov
'''

import numpy as np
import scipy.sparse as sps

### all parameters are in params_bio.py so we can import all of them:
from functions_bio import *
#########################################################################
def compute_yz(chi_1,chi_2,chi_3,a,beta,XN,params):
################################################################################
    omega=params['omega'];omega_v=params['omega_v'];s=params['s']
    Psi_S=params['Psi_S']
    #############################################
    # Psi_S0=Psi_S[0] ## used to check
    # print('Psi_S0=',Psi_S[0])
    #############################################
    eps_1=params['eps_all'][0]; eps_2=params['eps_all'][1]; eps_3=params['eps_all'][2]
# compute Y_{ij} and Z_{ij}
    nloc=np.shape(chi_1); mloc=nloc[1]; nloc=nloc[0]
    ntot=nloc*mloc
    D_Psi_S=sps.diags([Psi_S.reshape((ntot,))],[0],(ntot,ntot)) # this is diag(psi_S)
    d_s=sps.diags([s.reshape((nloc,))],[0],(nloc,nloc))
    d_beta=sps.diags([beta.reshape((nloc,))],[0],(nloc,nloc))
    s_a_beta=(d_s @ (a @ d_beta)).reshape(ntot,1)
    d_s_a_beta=sps.diags([s_a_beta.reshape(ntot,)],[0],(ntot,ntot))
    #print(d_s_a_beta)
    ch1=chi_1.reshape((ntot,1)); ch2=chi_2.reshape((ntot,1));ch3=chi_3.reshape((ntot,1))
    p1=ch1/(omega_v+eps_2*ch2)
    p2=(eps_2/omega)*ch2
    p3=1+(eps_3/omega)*ch3
    SN=(1+p1*(1+p2+p2*p3))
    SN=1./SN
    VN=p1*SN
    PN=p2*VN
    RN=p3*PN
#####    GN=(1.-eps_2)*VN+(1.-eps_3)*PN
    GN=(1.-eps_1)*VN+PN
    ## yz is the Schur product 
    D=(D_Psi_S @ d_s_a_beta)
    #  no need to reshape; work with sparse 
    y = (D @ SN) #.reshape((nloc,mloc))
    z = (D @ GN)#.reshape((nloc,mloc)) 
    return y,z
################################################################################
def compute_F(y,z,C,eta,params):
################################################################################
# compute F
    # print('|C|=',np.shape(eta),np.shape(C))
##    np.savetxt('C.dat',C)
    CC=sps.csr_matrix(C)
    # C4=C.reshape((4,4,4,4))
# C is ntotal by ntotal full matrix. 
    ntotal=np.shape(C)[0]
#
    Psi_I=params['Psi_I']
    D_Psi_I=sps.diags([Psi_I.reshape(ntotal,)],[0],(ntotal,ntotal))
    D_y=sps.diags([y.reshape(ntotal,)],[0],(ntotal,ntotal))
    D_z=sps.diags([z.reshape(ntotal,)],[0],(ntotal,ntotal))
    # D_y=D_y @ D_Psi_I
    # D_z=D_z @ D_Psi_I
    neta=np.shape(eta)[0]
    e1=sps.csr_matrix(np.array([1.]+(neta-1)*[0.]).reshape((neta,1)))
    e2=sps.csr_matrix(np.array([0.,1.]+(neta-2)*[0.]).reshape((neta,1)))
    FY0=(D_y @ CC) @ D_Psi_I
    FZ0=(D_z @ CC) @ D_Psi_I  
    
    row=-np.ones(2*neta*ntotal*ntotal,dtype=int)
    col=-np.ones(2*neta*ntotal*ntotal,dtype=int)
    vals=np.empty(shape=(2*neta*ntotal*ntotal,),dtype=float)
    count=0
    for k in range(2):
        for l in range(neta):
            etal=eta[l]
            for i in range(ntotal):
                for j in range(ntotal):
                    r=int(ntotal*i+j)
                    bl=not bool(r-(r//2)*2)
                    if(bl):
                        c=l+neta*k
                    else:
                        c=l+neta*k+neta*neta
                    row[count]=r//2
                    col[count]=c
                    if(k==0):
                        vals[count]=FY0[i,j]*etal 
                    else:
                        vals[count]=FZ0[i,j]*etal #F3[i,j,k,l]+F4[i,j,k,l]
                    count=count+1
    Fr = sps.coo_matrix((vals, (row, col)), shape=(ntotal*neta, ntotal*neta)).tocsr()
    return Fr
    
### MAIN ###
## setup parameters.
params=params_bio0()
## add two zeros to eta
params['eta']=np.concatenate((np.array([0.,0.]).reshape(2,),params['eta']),axis=0)
for key in params.keys():
    comm=key+"""=params['"""+key+"""']"""
    exec(comm,None,None)

# READ input matrices and vectors and set up the stage
dmat=read_all_data_files()
for key in dmat.keys():
    comm=key+"""=dmat['"""+key+"""']"""
    exec(comm,None,None)
del dmat
################ end input
##
#chi1=np.random.rand(4,4);chi2=np.random.rand(4,4);chi3=np.random.rand(4,4)
##
y,z=compute_yz(chi1,chi2,chi3,a,beta,XN,params)
#print("ey=",y-Y0,"\nez=",Z0-z)

# (i,p,j,q)-->(i,j,p,q)
C=C.transpose((0,2,1,3))
ndimC=np.shape(C) 
n=ndimC[0]
m=ndimC[1]
ntotal=n*m
C=C.reshape(ntotal,ntotal)

#ek0,vk0=calc_rho(C)
#print(vk0)
#exit(100)
#print("Shape(C)=",np.shape(C))
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
