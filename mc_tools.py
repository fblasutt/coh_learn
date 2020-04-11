# these are Markov chain tools
# note that this version does not utilize grid_combined class
import numpy as np
from numba import jit
from scipy.stats import norm

def mc_simulate(statein,Piin,shocks=None):
    # this simulates transition one period ahead for a Markov chain
    import numpy as np
    assert(np.max(statein) < Piin.shape[0] )
    assert statein.ndim == 1
    n_in = statein.size
    Picum = np.cumsum(Piin,axis=1)
    Pi_state = Picum[statein,:]
    
    assert np.all( np.abs(1 - Picum[:,-1]) < 1e-5) 
    
    #print(Pi_state)
    
    if shocks is None:
        rand = np.random.rand(n_in)[:,np.newaxis]
    else:
        rand = shocks[:,np.newaxis]
    
    rand_state = rand.repeat(Piin.shape[1],axis=1)
    stateout = np.zeros_like(statein)
    stateout[:] = np.sum((rand_state >= Pi_state),axis=1).squeeze() # note that numeration starts with 0
    assert(np.max(stateout) < Piin.shape[1])
    assert stateout.shape == statein.shape
    
    return stateout 

# yet another    
    
def mc_init_normal(sigma,X,N=1000,shocks=None):
    # this generates N draws from N(0,sigma^2)
    # then it looks at points at X that are the closest to 
    # these draws points and for each draw it returns its index
    import numpy as np
    if shocks is None:
        ran = sigma*np.random.normal(np.zeros(N))
    else:
        ran = sigma*shocks
    
    #X = x.reshape([x.size,1]).repeat(y.size,axis=1)
    #Y = y.reshape([1,y.size]).repeat(x.size,axis=0)
    
    
    return abs(ran[:,np.newaxis]-X).argmin(axis=1)

    

def trim_matrix(M,level=0.001):
    # this eliminates transition probabilities that are lower than level, 
    # renormalizing remaining probabilities to add up to one
    
    if isinstance(M,list):
        Mout = list()
        for m in M:
            Mout = Mout + [trim_one_matrix(m,level)]
    else:
        Mout = trim_one_matrix(M,level)
     
    return Mout

#@jit
def trim_one_matrix(M,level=0.001):
    
    Mout = M
    Mout[np.where(M<level)] = 0
    Mout = Mout / np.sum(Mout,axis=1)[:,np.newaxis]
    return Mout

def cut_matrix(M,kill_diag=False):
    # This routine takes matrix M as input and returns a 'cut' Matrix M'
    # all the entries states above the current one are made equal to
    # zero. 
    # Kill diag makes all the entries in the diagonal equal to zero.
 
    #Assert we have a aquare matrix
    assert np.shape(M)[0]==np.shape(M)[1]
    dim=np.shape(M)[0]
    
    MM=M.copy()
    #Kill the diagonal
    if kill_diag:
        for a in range(dim):
            if a>0:
                MM[a][a]=0.0
                
    #Fill the zeros
    for a in range(dim):
        for b in range(dim):
            if b>a:
                MM[a][b]=0.0
                
    #Rescale the non-zero entries
    for a in range(dim):
       
        MM[a]=np.array(MM[a])/sum(MM[a])
    
    return MM
    
    
def combine_matrices(a,b,Pia,Pib,check=True,trim=True,trim_level=0.00001):
    # this combines INDEPENDENT transition matrices Pia and Pib
    grid = mat_combine(a,b)
    
    Pi = np.kron(Pia,Pib) if ((Pia is not None) and (Pib is not None)) else None
    
    if Pi is not None:
        if check:
            assert(all(abs(np.sum(Pi,axis=1)-1)<1e-5))
        
        if trim:
            Pi = trim_matrix(Pi,trim_level)
    
    return grid, Pi
  
#@jit
def combine_matrices_list(alist,b,Pialist,Pib,check=True,trim=True,trim_level=0.00001):
    # this combines each element of Pialist and Pib
    # they assumed to be independent (i.e. Pialist and Pib can be combined in any order)
    grid, Pi = (list(), list())
        
    
    for i in range(0,len(Pialist)):
        gr_a, Pi_a = combine_matrices(alist[i],b,Pialist[i],Pib,check,trim,trim_level)
        grid = grid + [gr_a]
        Pi   = Pi + [Pi_a]
        
    if len(alist) > len(Pialist): # fix in case alist has one more element
        grid = grid + [mat_combine(alist[-1],b)]
        
    return grid, Pi


#@jit
def combine_matrices_two_lists(alist,blist,Pialist,Piblist,check=True,trim=True,trim_level=0.00001):
    # this combines each element of Pialist and Pib
    # they assumed to be independent (i.e. Pialist and Pib can be combined in any order)
    grid, Pi = (list(), list())
        
    assert len(alist) == len(blist)
    
    
    for i in range(0,len(Pialist)):
        gr_a, Pi_a = combine_matrices(alist[i],blist[i],Pialist[i],Piblist[i],check,trim,trim_level)
        grid = grid + [gr_a]
        Pi   = Pi + [Pi_a]
        
    if len(alist) > len(Pialist): # fix in case alist has one more element
        grid = grid + [mat_combine(alist[-1],blist[-1])]
        
    return grid, Pi
        
def combine_matrices_dependent(a,Pialist,b,Pib):
    # this assumes that there is unique transition matrix of a
    # corresponding to each value of b (so alist[j] is conditional transition
    # matrix of a given b = b[j])
    
    assert len(Pialist) == b.size, "Values in list should correspond to values in b"
    
    grid = mat_combine(b,a) # note the order
    
    n_a = a.size
    n_b = b.size
    
    n_ab = a.size*b.size
    
    
    Pi = np.zeros((n_ab,n_ab))
    
    for jb_from in range(n_b):
        for jb_to in range(n_b):
            Pi[jb_from*n_a:(jb_from+1)*n_a, jb_to*n_a:(jb_to+1)*n_a] = Pib[jb_from,jb_to]*Pialist[jb_from]
        
    
    assert(all(abs(np.sum(Pi,axis=1)-1)<1e-5))
    
    return grid, Pi
    



def mat_combine(a,b):
    # this gets combinations of elements of a and b
    
    a = a[:,np.newaxis] if a.ndim == 1 else a
    b = b[:,np.newaxis] if b.ndim == 1 else b
    
    assert a.ndim==2 and b.ndim==2
    
    l_a = a.shape[0]
    l_b = b.shape[0]
    
    w_a = a.shape[1]
    w_b = b.shape[1]
    
    grid = np.empty((l_a*l_b, w_a+w_b))
    
    for ia in range(l_a):
        grid[ia*l_b:(ia+1)*l_b,0:w_a] = a[ia,:] # this is broadcasting
        grid[ia*l_b:(ia+1)*l_b,w_a:(w_a+w_b)] = b
        
        
    return grid
    
def vec_combine(a,b):
    a_rep = a[:,np.newaxis].repeat(b.size,axis=1)
    b_rep = b[np.newaxis,:].repeat(a.size,axis=0)    
    grid = np.concatenate((a_rep.flatten()[:,np.newaxis],b_rep.flatten()[:,np.newaxis]),axis=1)
    return grid



def ind_combine(ia,ib,na,nb):
    return ia*nb + ib
    

def ind_combine_3(ia,ib,ic,na,nb,nc):
    return ia*nb*nc + ib*nc + ic
        
def ind_combine_m(ilist,nlist):
    out = 0
    for (pos,ind) in enumerate(ilist):
        out = np.int32( out + ind*np.prod(nlist[pos+1:]) ) 
    return out
    
    


def int_prob_standard(vec,trim=True,trim_level=0.001,n_points=None):
    # given ordered vector vec [x_0,...,x_{n-1}] this returns probabilities
    # [p0,...,p_{n-1}] such that p_i = P[d(Z,x_i) is minimal among i], where
    # Z is standard normal ranodm variable
    
    try:
        assert np.all(np.diff(vec)>0), "vec must be ordered and increasing!"
    except:
        print(vec)
        assert False
    
    vm = np.concatenate( ([-np.inf],vec[:-1]) )
    vp = np.concatenate( (vec[1:],[np.inf]) )
    v_up   = 0.5*(vec + vp)
    v_down = 0.5*(vec+vm)
    assert np.all(v_up>v_down)
    
    p = norm.cdf(v_up) - norm.cdf(v_down)
    
    
    
    if n_points is not None:
        trim = False # npoints overrides trim
        i_keep = ((-p).argsort())[0:n_points]
        pold = p.copy()
        p = np.zeros(p.shape)
        p[i_keep] = pold[i_keep]
        assert np.any(p>0)
        p = p / np.sum(p)
    
    if trim:
        p[np.where(p<trim_level)] = 0
        p = p / np.sum(p)
        
        
    
    
    #ap = np.argpartition(p,[-1,-2])
    assert(np.abs(np.sum(p) - 1) < 1e-8)
    
    return p#, ap, p[ap[-2]], p[ap[-1]]


def int_prob(vec,mu=0,sig=1,trim=True,trim_level=0.001,n_points=None):
    # this works like int_prob_standard, but assumes that Z = mu + sig*N(0,1)
    vec_adj = (np.array(vec)-mu)/sig
    return int_prob_standard(vec_adj,trim,trim_level,n_points)








# this tests combining things
if __name__ == "__main__":
    
    # teset combinations
    v0 = np.array([1,2,4])
    v1 = np.array([-1,-2])
    
    v_comb = np.array([[1,-1],[1,-2],[2,-1],[2,-2],[4,-1],[4,-2]])

    assert np.all(mat_combine(v0,v1) == v_comb)
    assert np.all(vec_combine(v0,v1) == mat_combine(v0,v1))
    
    
    # test transition probabilities
    
    vec = [-2,-1.8,-0.75,0,1,1.5,2]
    print(int_prob(vec,-1,1))
    
    
    i = ind_combine_m((1,2,4),(10,10,10))
    i2  = ind_combine_3(1,2,4,10,10,10)    
    print((i,i2))