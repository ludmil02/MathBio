import numpy as np
import scipy.sparse as sps
import re
def params_bio1():
    params={}
    m = 4; n = 4; ntotal=m*n
    x = 7.; 
    params['n']=n
    params['m']=m
    params['ntotal']=ntotal
# 
    params['x']=x
    params['kappa'] = 0.2*x; 
    params['xi'] = 0.5*x; 
    params['gamma_a'] = 1./7. 
    params['gamma_s'] = 0.2 
    params['gamma_m'] = 0.2 
    params['gamma_h'] = 0.1 
#    
    params['omega']=x/182.5
    params['omega_v']=params['omega']
# 
    params['eps_all']=np.array([0.8,0.8,0.8])
#
    params['bS'] = 0.31; 
    params['bI'] = 0.31; 
#    vgamma_a = [params['gamma_a'] k= {j, n}] x;
#    vgamma_s = [\[Gamma]s, {j, n}] x; 
#    vgamma_m = Table[\[Gamma]m, {j, n}] x; 
#    vgamma_h = Table[\[Gamma]h, {j, n}] x; 
# 
    params['beta'] = np.array([0.0794498883424254, 0.07378658226273424,0.08093055958811815,0.21470494228237624])
# 
    params['vp'] = np.array([0.9706574350846233, 0.8938517556772458, 0.6548278086229462,0.28889590656079006])
# 
    params['vq'] = np.array([0.9853287175423117, 0.9469258778386229, 0.8274139043114731,0.6444479532803951])
# 
    params['vdelta_s'] = np.array([0.0005227159753841133,0.010632200553688746,0.02953817543239884,0.057703228431405976])
# 
    params['vdelta_m'] = np.array([0.00026135798769205665,0.005316100276844373,0.01476908771619942,0.028851614215702988])
# 
    params['mN'] = np.array([[12861646, 14862307, 14294150, 13831766], 
                             [17039947, 18046593, 16847952, 16382252], 
                             [31775045, 34108261, 31760738, 29018710], 
                             [19577717, 22475544, 19374280, 17227215]],dtype=float)
# 
    params['ma'] = np.array([[17.073281356320802, 17.011021886542494, 14.464215476578978,3.6569593567943652], 
                             [13.676777723325813, 13.626903952047815, 11.586751010942738, 2.929455634349783], 
                             [15.06551341188455, 15.010575473629661, 12.763266047514328,3.2269116338380703], 
                             [13.603540223639197, 13.553933520407654, 11.524705352915733, 2.913768751778259]]);
    
    params['eta_a']=0.5
    params['eta_ps']=1.25 
    params['eta_s']=1.
    params['eta_h']=0.1 
    params['eta_pm']=1. 
    params['eta_m']=0.75
 #
# In this we need to follow the order used in the definition of F:

    params['eta']=np.array([params['eta_a'],params['eta_ps'],
                            params['eta_s'],params['eta_h'],params['eta_pm'], 
                            params['eta_m']]) ## later two zeros are added.
# 
    params['s'] = np.array([0.35861308333097464,0.4312100933817163, 0.4612003634870343,0.3916131469870049]).reshape(1,n)
# 
    Psi_S = np.array([0.7781979107138681,0.7798533321459905,0.7865615336962511,0.7801996496586996]).reshape(1,n)
    Psi_I= np.array([0.7781979107138681,0.7798533321459905,0.7865615336962511,0.7801996496586996]).reshape(1,n)

#     # const columns:
    params['Psi_S']=np.ones((n,1),dtype=float) @  Psi_S
    params['Psi_I']=np.ones((n,1),dtype=float) @  Psi_I
# 
    # params['chi1'] = np.array([[0.25,0.50, 0.16, 0.12], [0.14, 0.27, 0.095, 0.07], 
    #                  [0.17, 0.35, 0.12,0.09], [0.16, 0.31, 0.11, 0.08]]).reshape((ntotal,))
    # params['chi2'] = np.array([[0.002,0.007, 0.0006, 0.0003], 
    #                  [0.003, 0.012, 0.001, 0.0007], 
    #                  [0.006,0.024, 0.002, 0.001], 
    #                  [0.015, 0.054, 0.005, 0.003]]).reshape((ntotal,))
    # params['chi3'] = np.array([[0, 0, 0,0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],dtype=float).reshape((ntotal,));
    #ttt = 0.2*0.25/np.arctan(1.)
    #params['chi3'] = ttt*params['chi2'];
    return params
# EndOfFile
################################################################################
##########################################################################################
def read_matrix_csv(fname="nb_Data/Fmatrix.csv"):
    with open(fname, "r") as fp:
        s = fp.read()
    s="\n".join(s.splitlines()) # fix #r
    ## now split at \n as the csv has a new line separator.
    s=s.split("\n")
    Fall=[]
    for si in s:
        si = re.sub(r'''","''',"", si)
        si = re.sub(r"[\'\"]+", "", si)
        # Replace spaces, tabs, and newlines with an empty char
        si = re.sub(r"[\s\t\n]+", "", si)
        # print(si)
        k=0
        p0="\{"
        while(1):
            p=re.compile(p0)
            match = re.search(p, si)
            if match:
        #        print("Match at [", match.start(),"--",match.end(),"]")
                p0=p0+"\{";p=re.compile(p0);
                k=k+1
            else:
                break
        pb=k*'}'; pe=k*'{'
        ps=pb+pe
        if(k):
            row=si.split(ps)
            sjoin=pb+chr(171)+pe
            row=(sjoin.join(row)).split(chr(171))
        else:
            row='{'+si+'}'
            row=[row]
            # print('row=',row,flush=True)
        # print('row=',row,flush=True)
        F=[]; j=0
        for item in row:
            if(k):
                itemx=item
            else:
                # print(row[0])
                itemx=row[0]
            s1 = re.sub(r"\{", "[", itemx)
            s1 = re.sub(r"\}", "]", s1)
            s1 = re.sub(r"\*\^", "e", s1)
            s2="FX="+s1
            # print(s1,s2,flush=True)
            d = {}
            exec(s2,None,d)
            # z=get_shape(d['FX'])
            # print(z)
            F.append(d['FX'])
            # print(j,np.shape(np.array(FX,dtype=float)))
            j=j+1
            del s1
            del d
        Fall.append(F)
    Fall=np.array(Fall,dtype=float)
    # if(np.shape(Fall)[1]==1):
    #     Fall=Fall[:,0,:]
    return np.squeeze(Fall)
################################################################################
def read_data_files(data_files={},dir0='''nb_Data/''',ext='''.csv'''):
    dd={}
    for key in data_files.keys():
        d={}
        comm=key+'''=read_matrix_csv("'''+dir0+data_files[key]+ext+'''")'''
        #print(comm)
        exec(comm,None,d)
        dd[key]=d[key]
        del d
    return dd
################################################################################
def read_H_data_files(data_files,dir0='''nb_Data/''',ext='''.csv'''):
    dd={}
    for key in data_files.keys():
        d={}
        comm=key+'''=read_matrix_csv("'''+dir0+data_files[key]+ext+'''")'''
        #print(comm)
        exec(comm,None,d)
        dd[key]=d[key]
        del d
    return dd
################################################################################
def calc_rho(A,neig=8):
#     if(not sps.issparse(A)):
#         ee,vv=np.linalg.eig(A)
#     else:
#         print("*** using eigs for sparse F ***")
    ee,vv=sps.linalg.eigs(sps.csr_matrix(A), k=neig, which='LM',return_eigenvectors=True)
# order 
    perm = np.argsort(np.abs(ee))[::-1]  
    vv = vv[:, perm]
    ee=ee[perm]
# order is done; normalize the eigenvector for rho
    vv[:,0]=(1./np.linalg.norm(vv[:,0],ord=1))*vv[:,0]
#    print(np.max(abs(ee)),np.max(ee),ee[0])
#     # xis_real=np.linalg.norm(vv[:,0]-np.real(vv[:,0]))
#     # print('xis_real=',xis_real)
#     # print(perm[0:neig])
    print("ee=",ee[0:neig])
    ek,vk,flag=power_method(A,1e-12,iprt=True)
    if(flag):
        print('Power method did not converge for max num iters')
    else:
        pass
    ##ok normalize the vectors
    vk=(1./np.linalg.norm(vk,ord=1))*vk
    # print('chk0=',ek-ee[0])
    # print('chk1=',np.linalg.norm(abs(vk)-abs(vv[:,0]),ord=1))
    if(False):
       print('Density A:',A.nnz/(nF*mF))

    return ek,vk
import numpy as np
import scipy.sparse as sps
import re
################################################################################
# set parameters:
def params_bio0():
    params={}
    params['eps_all']=np.array([0.8,0.8,0.8])
    x=7. # many times used below, so we keep such local value
    params['x']=x

    params['omega']=x/182.5
    params['omega_v']=params['omega']
#
# OLD
    # params['eta_a']=0.5
    # params['eta_ps']=1.25
    # params['eta_s']=1.
    # params['eta_h']=0.1
    # params['eta_pm']=1. #was 0.5 now is 1.?
    # params['eta_m']=0.5
    #
    params['eta_a']=0.5
    params['eta_ps']=1.25 
    params['eta_s']=1.
    params['eta_h']=0.1 
    params['eta_pm']=1. 
    params['eta_m']=0.75
 #
# In this we need to follow the order used in the definition of F:

    params['eta']=np.array([params['eta_a'],params['eta_ps'],
                            params['eta_s'],params['eta_h'],params['eta_pm'], 
                            params['eta_m']]) ## later two zeros are added.
#
# from NB how it enters F:\[Eta]a  
#    \[Eta]ps 
#    \[Eta]s  
#    \[Eta]h  
#    \[Eta]pm  
#    \[Eta]m 
#

    #ord=['eta_a', 'eta_ps', 'eta_pm', 'eta_s', 'eta_m', 'eta_h']
    
    params['kappa'] = 0.2 * x; 
    params['xi']=0.5 * x
#
    params['P']=np.array([0.971, 0.894, 0.655, 0.289]) 
    params['Q']=np.array([0.985, 0.947, 0.827, 0.644])
#
    params['gamma_a']=x/7.
    params['gamma_s']=0.2 * x
    params['gamma_m']=0.2 * x
    params['gamma_h']=0.1 * x
    params['gamma_s_to_h']=0.5*x
    params['gamma_m_to_h']=params['gamma_s_to_h'] 
#
    params['delta_s']=0.5
    params['delta_m']=0.2
#
    # Psi_I = np.array([0.8225583285710945, 0.8238826657167924,0.8292492269570009, 0.8241597197269597]).reshape((1,4))
    # Psi_S = np.array([0.8225583285710945, 0.8238826657167924,0.8292492269570009, 0.8241597197269597]).reshape((1,4))
    # params['s'] = np.array([0.7024221010188523, 0.7349559739895175, 0.717797338653636,0.6535707127743224]);
    ## from eig_notebook
    Psi_S = np.array([0.7781979107138681, 0.7798533321459905,0.7865615336962511, 0.7801996496586996]).reshape((1,4))
    Psi_I = np.array([0.7781979107138681, 0.7798533321459905,0.7865615336962511, 0.7801996496586996]).reshape((1,4))
    params['s'] = np.array([0.35861308333097464, 0.4312100933817163, 0.4612003634870343, 0.3916131469870049]);
    nloc=4
    # Psi_S=np.array([0.823, 0.824, 0.829, 0.824]).reshape((1,4))
    # Psi_I=np.array([0.985, 0.947, 0.827, 0.644]).reshape((1,4))
    # const columns:
    params['Psi_S']=np.ones((nloc,1),dtype=float) @  Psi_S
    params['Psi_I']=np.ones((nloc,1),dtype=float) @  Psi_I
#    print([params['Psi_I'][0][q] for q in range(4)])
#    exit(5)
# const rows:
    #params['Psi_S']=Psi_S.T @ np.ones((nloc,1),dtype=float).T #const rows
    #params['Psi_I']=Psi_I.T @ np.ones((nloc,1),dtype=float).T 
    return params
################################################################################
################################################################################
def coord_lattice(index=int(-1),dim=int(4),nall=int(0),nd=np.array([0,0,0,0])):
# index is the global index
# nd is dimension dim and contains 
# the dimensions of the lattice in every direction
# output: m, i.e. the lattice indices from 0 to (dim-1)
    m=np.zeros(dim,dtype=int)
    j=np.prod(nd,dtype=int)  
    k=index;
    for i in range(dim, 0, -1):
        j=j//(nd[i-1])
        m[i-1] = k//j
        k=k - m[i-1]*j
    return m
#**********************************************************************
def num_lattice(lat=None,dim=int(4),nd=np.array([0,0,0,0],dtype=int)):
# returns global index for a node with lattice indices stored in m[0]...m[dim-1] 
# nd[] is the lattice size in each direction 
# dim is the dimension;  
##
    index=lat[dim-1]
    for i in range(dim-1, 0, -1):
        index=index*nd[i-1]+lat[i-1]
    return index
################################################################################
def power_method(A=np.random.rand(3,3),tol=1e-8,iprt=False):
    ##
    ## [yk,lambdak]=power_method(A,tol)
    ## power method for eigenvalues (tol=toleraance)
    ## it does not work for complex eigenvalues. 
    maxiter=1000
    n=np.shape(A)[0]
    if(not sps.issparse(A)):
        A=sps.csr_matrix(A,shape=(n,n))
    xk=np.random.rand(n,1)
    iter=0
    if(iprt):
        print('max_iter=',maxiter,"; tol=",tol)
        while(1):
            print(xk)
            yk=(1./np.linalg.norm(xk))*xk
            xk=A @ yk
            ek=yk.T @ xk ##
            ## check convergence
            stop0=np.linalg.norm(xk-ek*yk); ##
            print('power_method_iter=',iter,' ek=',ek,' |A*y-lambda * y|=', stop0)
            iter=iter+1
            if(stop0<tol):
                flag=0
                break
            if(iter>maxiter): 
                flag=1
                break
    else:
        while(1):
            print(xk)
            yk=(1./np.linalg.norm(xk))*xk
            xk=A @ yk
            ek=yk.T @ xk ##
            # check convergence
            stop0=np.linalg.norm(xk-ek*yk); ##
            iter=iter+1
            if(stop0<tol):
                flag=0
                break
            if(iter>maxiter):
                flag=1 
                break
    return np.squeeze(ek),np.squeeze(xk),flag
