import numpy as np
import networkx as nx
import random

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

def vertex_with_most_edges(graph):
    degrees = dict(graph.degree())     
    max_degree = max(degrees.values())
    max_degree_vertices = [vertex for vertex, degree in degrees.items() if degree == max_degree]

    return max_degree_vertices, max_degree

def supervised_gromov_wasserstein(
    D1: np.ndarray, 
    D2: np.ndarray,
    eps: float = 1e-2,
    nitermax: int = 10000, 
    threshold: float = 0.02,
    random_state: int = 1,
    verbose: bool=False
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
        Whether to print intermediate information.
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
  
    tuples_set = set()
    B = G.copy()

    while B.edges:
        vertices, degree = vertex_with_most_edges(B)
        vertex = random.choice(vertices)
        B.remove_node(vertex)
        tuples_set.add(vertex)

    if not B.edges and verbose:
        print("No more edges in the graph.")  
    if verbose:
        print(f"Minimum vertex cover algorithm is done, which finds {len(tuples_set)} out of {m * n} entries are 0 in the coupling P!")


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