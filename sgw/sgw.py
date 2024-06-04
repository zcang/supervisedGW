import numpy as np
import networkx as nx
import random


def mvc_exp(C1, C2, n_exp=10, value=1, penalty=10, niter_sOT=10000, threshold_min=0.0001,                       threshold_max=0.5000, save_path='', save_results=False):
    rhos = np.full(n_exp, value)
    ratios = np.random.uniform(threshold_min, threshold_max, n_exp)
    a = np.ones([C1.shape[0]]) / C1.shape[0]
    b = np.ones([C2.shape[0]]) / C2.shape[0]
    sot_options = {'penalty': penalty, 'niter_sOT': niter_sOT, 'f_init': np.zeros_like(a), 'g_init':           np.zeros_like(b)}
    A_sum_list = []
    P_sum_list = []
    Ratio_list = []
    A_matrices_list = []

    for i in range(n_exp):
        A, test = get_cover_4o(C1, C2, threshold=rhos[i], ratio=ratios[i])
        if test == 0:
            print("error")
            break
        
        if test == 1:
            C = np.ones_like(A)
            C[np.where(A > 0)] = np.inf
            P, _, _ = perform_sOT_log(C, a, b, 0.01, sot_options)
            print(i, int(A.sum()), P.sum(), ratios[i])
            A_sum_list.append(int(A.sum()))
            P_sum_list.append(P.sum())
            Ratio_list.append(ratios[i])
            A_matrices_list.append(A)
    
    if save_results:
        A_matrices_tensor = np.array(A_matrices_list)
        np.save(save_path + 'synthetic_matrices.npy', A_matrices_tensor)
        np.save(save_path + 'synthetic_A.npy', A_sum_list)
        np.save(save_path + 'synthetic_P.npy', P_sum_list)
        np.save(save_path + 'synthetic_Ratio.npy', Ratio_list)
    
    if P_sum_list:
        # Find the indices of the maximum value in P_sum_list
        max_P_sum_value = np.max(P_sum_list)
        max_indices = np.where(P_sum_list == max_P_sum_value)[0]

        # Among these indices, randomly select one index
        min_ratio_index = random.choice(max_indices)

        # Get the corresponding matrix A
        corresponding_A_matrix = A_matrices_list[min_ratio_index]

        # Find the nonzero locations of the matrix A
        nonzero_indices = np.argwhere(corresponding_A_matrix > 0)

        # Convert the nonzero indices to a set of tuples representing the vertex cover
        tuples_set = set(tuple(idx + 1) for idx in nonzero_indices)
        
        print(tuples_set)
        return tuples_set
    else:
        print("No valid experiments were found.")
        return None

def vertex_with_most_edges(B):
    max_degree = max(dict(B.degree()).values())
    vertices = [v for v, d in B.degree() if d == max_degree]
    return vertices, max_degree

def is_vertex_cover(G, cover):
    for u, v in G.edges():
        if u not in cover and v not in cover:
            return False
    return True

def is_minimal_vertex_cover(G, cover):
    if not is_vertex_cover(G, cover):
        return False
    for vertex in list(cover):
        cover.remove(vertex)
        if is_vertex_cover(G, cover):
            return False
        cover.add(vertex)
    return True

def get_cover_4o(
    D1: np.ndarray, 
    D2: np.ndarray,
    ratio: float = 0.0001,
    threshold: float = 0.02,
    random_state: int = 1,
    verbose: bool=False
):    
    np.random.seed(random_state)
    random.seed(random_state)

    n = D1.shape[0]
    m = D2.shape[0]
  
    # Build graph
    G = nx.Graph()

    # Efficient tensor4 calculation
    C1_exp = D1[:, None, :, None]
    C2_exp = D2[None, :, None, :]
    tensor4 = (C1_exp - C2_exp) ** 2
    positions = np.where(tensor4 > threshold ** 2)
    
    G.add_edges_from(((p[0] + 1, p[1] + 1), (p[2] + 1, p[3] + 1)) for p in zip(*positions))
    
    tuples_set = set()
    B = G.copy()
    initial_random_add = int(G.number_of_nodes() * ratio)
    
    for i in range(initial_random_add):
        vertex = random.choice(list(B.nodes))
        B.remove_node(vertex)
        tuples_set.add(vertex)

    while B.edges:
        vertices, degree = vertex_with_most_edges(B)
        vertex = random.choice(vertices)
        B.remove_node(vertex)
        tuples_set.add(vertex)

    inf_mask = np.zeros([n, m])
    test = 0

    if is_minimal_vertex_cover(G, tuples_set):
        test = 1
        for (i, j) in tuples_set:
            inf_mask[i-1, j-1] = 1
    else:
        minimal_cover = tuples_set.copy()
        for vertex in tuples_set:
            minimal_cover.remove(vertex)
            if not is_vertex_cover(G, minimal_cover):
                minimal_cover.add(vertex)
        if is_minimal_vertex_cover(G, minimal_cover):
            test = 1
            for (i, j) in minimal_cover:
                inf_mask[i-1, j-1] = 1

    return inf_mask, test



def perform_sOT_log(G, a, b, eps, options):

    niter = options['niter_sOT']
    f     = options['f_init']
    g     = options['g_init']
    M     = options['penalty']

    # Err = np.array([[1, 1]])

    for q in range(niter):   
        f = np.minimum(eps * np.log(a) - eps * np.log(np.sum(np.exp((f[:, None] + g[None, :]  - G) / eps), axis=1)+ 10**-20) + f, M)
        g = np.minimum(eps * np.log(b) - eps * np.log(np.sum(np.exp((f[:, None] + g[None, :]  - G) / eps), axis=0)+ 10**-20) + g, M)

    P = np.exp((f[:, None] + g[None, :] - G) / eps)
    
    return P, f, g


def supervised_gromov_wasserstein(
    D1: np.ndarray, 
    D2: np.ndarray,
    eps: float = 1e-2,
    nitermax: int = 20, 
    threshold: float = 0.02,
    random_state: int = 1,
    verbose: bool=False,
    mvc_exp_params: dict = None
):
    """
    Solve the supervised Gromov-Wasserstein problem given two distance matrices

    Parameters
    ----------
    D1: shape (ns, ns)
        Metric cost matrix in the source space.
    D2: shape (nt, nt)
        Metric cost matrix in the target space.
    eps
        The coefficient for the entropy regularization term.
    nitermax
        The maximum number of iterations.
    threshold
        The threshold to exclude a pair of mapping. That is, :math:`\mathbf{P}_{ij}\mathbf{P}_{kl}=0`: if :math:`|\mathbf{D}_1(i,k)-\mathbf{D}_2(j,l)| > threshold`:.
    random_state
        The random state to use for reproducing results.
    verbose
        Whether to involve mvc_exp.
    mvc_exp_params
        Parameters for the mvc_exp function if verbose is True.
    """
    
    np.random.seed(random_state)
    random.seed(random_state)

    n = D1.shape[0]
    m = D2.shape[0]
    C1 = D1.copy()
    C2 = D2.copy()
  
    #build graph and find an approximation of min vertex covering
    G = nx.Graph()
    for i in range(n):
        for j in range(m):
            vertex = (i+1, j+1) 
            G.add_node(vertex)

    tensor4 = np.zeros((n, m, n, m))

    C1_exp = C1[:, None, :, None]
    C2_exp = C2[None, :, None, :]
    tensor4 = (C1_exp - C2_exp)**2
    
    positions = np.where(tensor4 > threshold**2) 
    vertex_positions = positions[0] + 1, positions[1] + 1, positions[2] + 1,positions[3] + 1

    for i in range(len(positions[0])) :
        first_elements = [arr[i-1] for arr in vertex_positions]
        vertex1 = (first_elements[0],first_elements[1])
        vertex2 = (first_elements[2],first_elements[3])
        G.add_edge(vertex1, vertex2) 
  
    if verbose and mvc_exp_params:
        tuples_set = mvc_exp(C1, C2, **mvc_exp_params)
    else:
        tuples_set = set()
        B = G.copy()

        while B.edges:
            vertices, degree = vertex_with_most_edges(B)
            vertex = random.choice(vertices)
            B.remove_node(vertex)
            tuples_set.add(vertex)

        if not B.edges:
            print("No more edges in the graph.")  


    #run sot and update P

    a = np.array([1 / n] * n, dtype=float)
    aa = a + 1e-1 * np.random.rand(n)
    b = np.array([1 / m] * m, dtype=float)
    bb = b + 1e-1 * np.random.rand(m)

    aa = aa / np.linalg.norm((aa), ord=1)
    bb = bb / np.linalg.norm((bb), ord=1)

    P = np.outer(aa, bb) 
    f = np.zeros(n)
    g = np.zeros(m)


    for p in range(nitermax):
        
        D = np.zeros((n, m)) 
        P_reshaped = P.reshape(1, 1, n, m)
        D = np.sum(tensor4 * P_reshaped, axis=(-2, -1))
                    
        for s, t in tuples_set:
            D[s-1, t-1] = np.inf
        
        options = {
        'niter_sOT': 10**5,
        'f_init': np.zeros(n),
        'g_init': np.zeros(m),
        'penalty': 2
        }
        
        P,f,g = perform_sOT_log(D, a, b, eps, options)
            
    return P

# Example usage
# mvc_exp_params = {'n_exp': 10, 'value': 1, 'penalty': 10, 'niter_sOT': 10000, 'threshold_min': 0.0001, 'threshold_max': 0.5000, 'save_path': '', 'save_results': False}
# P = supervised_gromov_wasserstein(D1, D2, verbose=True, mvc_exp_params=mvc_exp_params)