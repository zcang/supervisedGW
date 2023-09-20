import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import distance_matrix
from geosketch import gs
import kmapper as km
import umap
import sklearn

def sot_sinkhorn_l1_sparse(a,b,C,eps,m,nitermax=10000,stopthr=1e-8,verbose=False):
    """ Solve the unnormalized optimal transport with l1 penalty in sparse matrix format.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) scipy.sparse.coo_matrix
        The cost matrix in coo sparse format. The entries exceeds the cost cutoff are omitted. The naturally zero entries should be explicitely included.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.

    Returns
    -------
    (ns,nt) scipy.sparse.coo_matrix
        The optimal transport matrix. The locations of entries should agree with C and there might by explicit zero entries.
    """
    tmp_K = C.copy()
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    r = np.zeros_like(a)
    s = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err > stopthr:
        fprev = f
        gprev = g
        # Iteration
        tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
        f = eps * np.log(a) \
            - eps * np.log( np.sum( tmp_K, axis=1 ).A.reshape(-1) \
            + np.exp( ( -m + f ) / eps ) ) + f
        tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
        g = eps * np.log(b) \
            - eps * np.log( np.sum( tmp_K, axis=0 ).A.reshape(-1) \
            + np.exp( ( -m + g ) / eps ) ) + g
        # Check relative error
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter = niter + 1

    if verbose:
        print('Number of iterations in unot:', niter)
    tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
    return tmp_K

def recover_full_coupling(
    P, # n1_sub x n2_sub
    D1, # n1 x n1_sub
    D2, # n2 x n2_sub
    delta: float = 0.5,
    eps: float = 1e-2,
    thresh: float = 0.02,
    nitermax: int = 1e5,
    stopthr: float = 1e-8,
    p_cost = 2
):
    """
    Recover the optimal coupling between the two full datasets where sGW has been computed for subsampled data.

    Parameters
    ----------
    P
        The coupling matrix of shape ``n1_sub x n2_sub`` where the two dimensions are the number of subsampled points in the two datasets.
    D1
        The geodesic distance matrix in dataset 1 of shape ``n1 x n1_sub`` between full dataset and subsampled dataset.
    D2
        The geodesic distance matrix in dataset 2 of shape ``n2 x n2_sub`` between full dataset and subsampled dataset.
    delta
        The smallest fraction of transported mass for each subsampled point to be used as an "anchor" point.
    eps
        The coefficient of the entropy regularizer for the SOT solver.
    thresh
        The cost threshold for the SOT solver.
    nitermax
        Maximum iterations for the SOT solver.
    stopthr
        Stopping criteria for the SOT solver.
    p_cost
        The l_p distance used to compute the cost.
    """
    n1, n1_sub = D1.shape
    n2, n2_sub = D2.shape

    a = np.ones(n1) / n1
    b = np.ones(n2) / n2

    idx_1 = np.where(P.sum(axis=1) >= delta / n1_sub)[0]
    idx_2 = np.where(P.sum(axis=0) >= delta / n2_sub)[0]

    P_sig = P[idx_1,:][:,idx_2]
    D1_sig = D1[:,idx_1]
    D2_sig = D2[:,idx_2]
    P_sig_1_to_2 = P_sig.copy() / P_sig.sum(axis=1).reshape(-1,1)
    P_sig_2_to_1 = ( P_sig.copy() / P_sig.sum(axis=0).reshape(1,-1) ).T
    D1_to_D2 = np.matmul(D1_sig, P_sig_1_to_2)
    D2_to_D1 = np.matmul(D2_sig, P_sig_2_to_1)

    C1 = distance_matrix(D1_sig, D2_to_D1, p=p_cost)
    C2 = distance_matrix(D1_to_D2, D2_sig, p=p_cost)
    C = 0.5 * ( C1 + C2 )
    C += 1e-10
    C[np.where(C > thresh)] = 0
    C_sparse = sparse.coo_matrix(C)
    C_sparse.data -= 1e-10
    C_sparse = C_sparse / C_sparse.max()
    m = 2

    P_full = sot_sinkhorn_l1_sparse(a, b, C_sparse, eps, m, nitermax=nitermax, stopthr=stopthr)
    
    return P_full


def downsample_data(
    X: np.ndarray,
    method: str = 'geosketch', 
    random_state: int = 1,
    gs_N: int = 50,
    gs_replace: bool = False, # This means whether draw multiple samples from the same covering box.
    gs_k = 'auto',
    gs_alpha: float = 0.1,
    gs_max_iter: int = 200,
    km_proj_method: str='self', # 'self', 'top_dim', 'umap'
    km_proj_dim: int = 2,
    km_projection = None,
    km_n_cubes = 10,
    km_perc_overlap = 0.2,
    km_dbscan_eps = 0.5,
    km_dbscan_min_sample = 5,
    km_centroid_dimred = 'self',
):
    if method == 'geosketch':
        idx = gs(X, 
            gs_N, 
            replace=gs_replace, 
            k=gs_k, 
            alpha=gs_alpha, 
            max_iter=gs_max_iter,
            seed=random_state)
    elif method == 'mapper':
        mapper = km.KeplerMapper(verbose=0)
        if km_proj_method == 'self':
            projection = [i for i in range(X.shape[1])]
        elif km_proj_method == 'top_dim':
            projection = [i for i in range(km_proj_dim)]
        elif km_proj_method == 'umap':
            projection = umap.UMAP(n_components=km_proj_dim)
        elif km_proj_method == 'user':
            projection = km_projection
        projected_data = mapper.fit_transform(X, projection=projection)
        cover = km.Cover(n_cubes=km_n_cubes, perc_overlap=km_perc_overlap)
        graph = mapper.map(projected_data, cover=cover, clusterer=sklearn.cluster.DBSCAN(eps=km_dbscan_eps, min_samples=km_dbscan_min_sample))
        
        idx = []
        if km_centroid_dimred == 'self':
            XX = X.copy()
        elif km_centroid_dimred == 'projection':
            XX = projected_data.copy()
        for node in graph['nodes']:
            tmp_idx = graph['nodes'][node]
            tmp_X = XX[tmp_idx,:]
            tmp_D = distance_matrix(tmp_X, tmp_X)
            idx.append(tmp_idx[np.argmin(tmp_D.sum(axis=1))])

    return idx