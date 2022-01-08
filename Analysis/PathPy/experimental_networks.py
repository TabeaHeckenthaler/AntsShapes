from Analysis.PathPy.network_functions import *
from Analysis.States import States
from Analysis.PathPy.AbsorbingMarkovChain import *
import pandas as pd


def pathpy_network() -> (pp.Paths, pp.Network):
    # create paths and network
    paths = create_paths([states.state_series for states in list_of_states])
    n = pp.Network.from_paths(paths)
    plot_network(n)

    state_order = ['b', 'a', 'd', 'e', 'f', 'g', 'j']
    # state_dict = {v: n.node_to_name_map()[v] for v in state_order if v in n.node_to_name_map().keys()}
    return paths, n


def absorbing_state_analysis(T) -> np.array:
    m = transposeMatrix(T)
    m = sort(m)
    # norm = normalize(m)
    trans = num_of_transients(m)
    Q, R = decompose(m)
    P = np.vstack([np.hstack([identity(1), np.zeros([1, trans])]),
                   np.hstack([R, Q])])  # canonical form of transition matrix
    N = np.linalg.inv(identity(len(Q[-1])) - Q)  # fundamental matrix
    t = np.matmul(N, np.ones(N.shape[0]))  # expected number of steps before absorption from each steps
    B = np.matmul(N, R)  # absorption probabilities
    P_inf = np.linalg.matrix_power(P, 100)  # check whether P_inf is indeed np.array([[I, 0], [B, 0]])
    return t


def Markovian_analysis(n) -> np.array:
    # adjacency_matrix, degrees, laplacian, transition matrix and eigenvector
    A = n.adjacency_matrix().toarray()
    D = np.eye(len(n.degrees())) * np.array(n.degrees())
    L = n.laplacian_matrix(n.degrees()).toarray()
    T = n.transition_matrix().toarray()
    plot_transition_matrix(T, list(n.node_to_name_map().keys()) + ['i'])

    EV = n.leading_eigenvector(n.adjacency_matrix())
    return T


def higher_order_networks(paths):
    hon = create_higher_order_network(paths)
    hon.likelihood(paths)
    eigenvalue_gap = pp.algorithms.spectral.eigenvalue_gap(hon)

    # How much slower is the second order network than the markovian network?
    pp.algorithms.path_measures.slow_down_factor(paths)

    # slow down factor
    # Ratios smaller than one indicate that the temporal network exhibits non - Markovian characteristics
    pp.algorithms.path_measures.entropy_growth_rate_ratio(paths, method='Miller')

    # Is the process Markovian?
    # estimate the order of the sequence (something like memory)
    ms = pp.MarkovSequence(paths.sequence())
    order = ms.estimate_order(4)


if __name__ == '__main__':
    #
    # solver = 'ant'
    # sizes = ['XL', 'L', 'M', 'S']

    solver = 'human'
    sizes = ['Large']
    shape = 'SPT'

    results = pd.DataFrame()
    for size in sizes:
        conf_space_labeled = load_labeled_conf_space(solver=solver, size=size, shape=shape)

        trajectories = get_trajectories(solver=solver, size=size, shape=shape, number=20)

        list_of_states = [States(conf_space_labeled, x, step=int(x.fps/2)) for x in trajectories]

        paths, n = pathpy_network()
        T = Markovian_analysis(n)
        t = absorbing_state_analysis(T)
        series = pd.Series([t], ['t'], name=size)
        results = results.append(series)
    results.tojson(path.join(network_dir, 'Markov_results.json'))


# TODO: Look at returning to dead ends to define

