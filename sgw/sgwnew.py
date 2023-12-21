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
############################################
def vertex_with_most_edges(graph):
    degrees = dict(graph.degree())     
    max_degree = max(degrees.values())
    max_degree_vertices = [vertex for vertex, degree in degrees.items() if degree == max_degree]

    return max_degree_vertices, max_degree

###############################################
def edge_vc(G):
    """
    Approximate the minimum vertex cover for a graph.

    :param G: A NetworkX graph
    :return: A set representing the approximate minimum vertex cover
    """
    # Make a copy of the graph so we don't modify the original
    G_copy = G.copy()
    vertex_cover = set()

    # Keep removing edges and adding their vertices to the cover
    while G_copy.edges:
        # Select an edge
        edge = next(iter(G_copy.edges))
        u, v = edge

        # Add the vertices to the vertex cover
        vertex_cover.add(u)
        vertex_cover.add(v)

        # Remove all edges incident to these vertices
        G_copy.remove_node(u)
        G_copy.remove_node(v)

    return vertex_cover

###############################################

def edge_vc_random(G):
    """
    Approximate the minimum vertex cover for a graph with a randomized edge algorithm.

    :param G: A NetworkX graph
    :return: A set representing the approximate minimum vertex cover
    """
    G_copy = G.copy()
    vertex_cover = set()

    while G_copy.edges:
        # Randomly select an edge
        edge = random.choice(list(G_copy.edges))
        u, v = edge

        # Add the vertices to the vertex cover
        vertex_cover.add(u)
        vertex_cover.add(v)

        # Remove all edges incident to these vertices
        G_copy.remove_node(u)
        G_copy.remove_node(v)

    return vertex_cover
##############################################

def vertex_vc(G, ratio):
    
    """
     Approximate the minimum vertex cover for a graph based on vertex degree.
     Pre step is deleting without considering degree.

    :param G: A NetworkX graph
    :return: A set representing the approximate minimum vertex cover
    """   
    B = G.copy()
    vertex_cover = set()
    
    pre_delete = int(B.number_of_nodes()/ratio)
    
    for i in range(pre_delete):
        vertex = random.choice(list(B.nodes()))
        B.remove_node(vertex)
        vertex_cover.add(vertex)
        
    while B.edges:
        vertices, degree = vertex_with_most_edges(B)
        vertex = random.choice(vertices)
        B.remove_node(vertex)
        vertex_cover.add(vertex)
        
    return vertex_cover

#######################################
def vertex_vc_greedy(G):
    
    """
    Approximate the minimum vertex cover for a graph with a randomized greedy vertex degree algorithm.
    No pre step
    
    :param G: A NetworkX graph
    :return: A set representing the approximate minimum vertex cover
    """   
    B = G.copy()
    vertex_cover = set()
        
    while B.edges:
        vertices, degree = vertex_with_most_edges(B)
        vertex = random.choice(vertices)
        B.remove_node(vertex)
        vertex_cover.add(vertex)
        
    return vertex_cover

#########################################################
# define the infinity mask according to largest matrix sum of different vertex_covers
def inf_mask(G, n, m, eps, gamma):
    """
    Run different methods over different approximated vertex covers 30 times each 
    to get the one with the largest transported mass.
    """

    a = np.array([1 / n] * n, dtype=float)
    aa = a + 1e-1 * np.random.rand(n)
    b = np.array([1 / m] * m, dtype=float)
    bb = b + 1e-1 * np.random.rand(m)

    aa = aa / np.linalg.norm((aa), ord=1)
    bb = bb / np.linalg.norm((bb), ord=1)

    P = np.outer(aa, bb) 
    f = np.zeros(n)
    g = np.zeros(m)
    
    
    D = np.ones((n, m))
    max_matrix_sum = -np.inf
    best_vertex_cover = None
    best_method = None

    methods = {
        'edge_vc': edge_vc, 
        'edge_vc_random': edge_vc_random, 
        'vertex_vc_greedy': vertex_vc_greedy,
        'vertex_vc_16': lambda G: vertex_vc(G, 2**4),
        'vertex_vc_8': lambda G: vertex_vc(G, 2**3),
        'vertex_vc_4': lambda G: vertex_vc(G, 2**2),
        'vertex_vc_2': lambda G: vertex_vc(G, 2**1)
    }

    for method_name, method in methods.items():
        for _ in range(20):
            D.fill(1)  # Reset D matrix for each run
            for s, t in method(G):
                D[s-1, t-1] = np.inf

            options = {
                'niter_sOT': 5*(10**5),
                'f_init': np.zeros(n),
                'g_init': np.zeros(m),
                'penalty': gamma
            }

            P, f, g = perform_sOT_log(D, a, b, eps, options)
            matrix_sum = np.sum(P)

            if matrix_sum > max_matrix_sum:
                max_matrix_sum = matrix_sum
                best_vertex_cover = method(G)  # Assuming method(G) returns the vertex cover
                best_method = method_name

    return best_vertex_cover, max_matrix_sum, best_method

# Call the function with your graph G
# vertex_cover, P_sum, method_used = inf_mask(G)


#######################################################
#######################################################    
    
    
    

def supervised_gromov_wasserstein_new(
    D1: np.ndarray, 
    D2: np.ndarray,
    eps: float = 1e-2,
    nitermax: int = 10000, 
    threshold: float = 0.02,
    gamma: int = 2,
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
  
    # use the infinity mask that has the largest P_sum
    tuples_set, P_sum, method_used = inf_mask(G, n, m, eps, gamma)
    
    print(f"Inf_mask finds {len(tuples_set)} out of {n * m} entries are 0 in the coupling P!")
    print(f"Maximum Transported Mass (P_sum): {P_sum}")
    print(f"Method Used: {method_used}")
    
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
        'penalty': gamma
        }
        
        P,f,g = perform_sOT_log(D, a, b, eps, options)
            
    return P