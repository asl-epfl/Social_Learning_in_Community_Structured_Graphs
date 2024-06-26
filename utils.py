import numpy as np
from sinkhorn_knopp import sinkhorn_knopp as skp
from sklearn.cluster import KMeans


def create_network(agents, option, p=0.5):
    """
    :int agents: number of agents in the network
    :int option:
        1:Random (undirected)
        2:Fully connected
        3:Erdos-Renyi
        4:Star
    :float p: probability for ER model, otherwise use 0.5
    :return: np.array(agents, agents) adjacency matrix
    """
    # Random construction (undirected graph)
    if option == 1:
        option = 3
        p = 0.5
    # Fully connected
    if option == 2:
        adj_matrix = np.ones((agents, agents))
    # Erdos-Renyi
    elif option == 3:
        adj_matrix = np.random.rand(agents, agents)
        adj_matrix[adj_matrix < 1 - p] = 0
        adj_matrix[adj_matrix >= 1 - p] = 1
        # Symmetrization
        adj_matrix = np.tril(adj_matrix) + np.tril(adj_matrix).T
        np.fill_diagonal(adj_matrix, 1)
        return adj_matrix
    # Star
    elif option == 4:
        adj_matrix = np.zeros((agents, agents))
        center_node = 0
        adj_matrix[center_node, :] = 1
        adj_matrix[:, center_node] = 1
        np.fill_diagonal(adj_matrix, 1)
    # 2-Star
    elif option == 5:
        adj_matrix = np.zeros((agents, agents))
        adj_matrix[0,2:] = 1
        adj_matrix[1,2:] = 1
        adj_matrix[2:,0] = 1
        adj_matrix[2:,1] = 1
    elif option == 6:
        if agents != 18:
            raise ValueError("agents = 18.")
        adj_matrix = np.zeros((agents, agents))
        adj_matrix[0, :] = np.array([0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0])
        adj_matrix[1, :] = np.array([0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1])
        adj_matrix[7, 13], adj_matrix[7, 17] = 1, 1
        adj_matrix[8, 9], adj_matrix[8, 10], adj_matrix[8, 14] = 1, 1, 1
        adj_matrix[9, 10], adj_matrix[9, 14] = 1, 1
        adj_matrix[10, 11] = 1
        adj_matrix[15, 16] = 1
        ##
        adj_matrix[2, 4] = 1
        adj_matrix[3, 13] = 1
        adj_matrix[4, 17] = 1
        adj_matrix[6, 16] = 1
        ##
        adj_matrix = adj_matrix + adj_matrix.T
        adj_matrix[1, 1] = 1
    else:
        raise ValueError("Invalid selection of the option.")
    return adj_matrix


def perturbe_network(adj_matrix, p_change=0.05):
    change = np.random.rand(adj_matrix.shape[0], adj_matrix.shape[1])
    change[change<1-p_change] = 0
    change[change>0] = 1
    change = np.tril(change, 1)
    adj_matrix = adj_matrix + change
    adj_matrix[adj_matrix>1] = 0
    # Symmetrization
    adj_matrix = np.tril(adj_matrix) + np.tril(adj_matrix).T
    np.fill_diagonal(adj_matrix, 1)
    return adj_matrix


def generate_combination_weights(adj_matrix, option):
    """
    :np.array adj_matrix:
    :int option:
        0:Uniform
        1:Doubly stochastic
        2:Random (Left stochastic)
    :return: np.array combination_weights, np.array centrality, bool strongly_connected
    """
    weight = adj_matrix.copy()
    # Uniform
    if option == 0:
        weight = weight / weight.sum(0)[None, :]
    elif option == 1:
        sk = skp.SinkhornKnopp()
        weight = sk.fit(weight)
    elif option == 1:
        weight = np.random.uniform(size=weight.shape)
        weight[adj_matrix == 0] = 0
        weight = weight / weight.sum(0)[None, :]
    else:
        print("Invalid selection of the option.")

    e_values, e_vectors = np.linalg.eig(weight)
    e_index = np.argmax(e_values)
    centrality = e_vectors[:, e_index] / e_vectors[:, e_index].sum()
    strongly_connected = np.all(centrality > 0)
    return weight, centrality, strongly_connected


def create_likelihoods(agents, states, strategy, params=2, max_mean=10., max_std=3., var=.1, num_inf=3, state_true=None):
    """
    :int agents: number of agents in the network
    :int states: number of states in the network
    :int strategy:
        0:discrete manual
        1:discrete
        2:gaussian
        3:discrete manual with influencers
    :int params: number of different discrete states for 0 strategy
    :return: np.array(agents, states, params) likelihood matrix
    """
    if strategy == 0:
        probabilities = np.random.rand(agents, params)
        random_error = var * np.random.rand(agents, states, params)
        likelihood_matrix = np.repeat(probabilities[:, np.newaxis, :], states, axis=1) + random_error
        likelihood_matrix = likelihood_matrix / likelihood_matrix.sum(2)[:, :, None]
    elif strategy == 1:
        likelihood_matrix = np.random.rand(agents, states, params)
        likelihood_matrix = likelihood_matrix / likelihood_matrix.sum(2)[:, :, None]
    elif strategy == 2:
        params = 2
        likelihood_matrix = np.random.rand(agents, states, params)
        likelihood_matrix[:, :, 0] *= max_mean
        likelihood_matrix[:, :, 1] *= max_std
        likelihood_matrix = likelihood_matrix / likelihood_matrix.sum(2)[:, :, None]
    elif strategy == 3:
        probabilities = .3 * np.ones((agents, params))
        probabilities[:, 1:] = .7 * np.ones((agents, params-1))
        random_error = np.zeros((agents, states, params))
        for s in range(states):
            random_error[num_inf:, s, 0] += 0.05 * s * np.ones((agents - num_inf))
            random_error[:num_inf, s, 0] += 0.5 * s * np.ones((num_inf))
        likelihood_matrix = np.repeat(probabilities[:, np.newaxis, :], states, axis=1) + random_error
        likelihood_matrix = likelihood_matrix / likelihood_matrix.sum(2)[:, :, None]
    elif strategy == 4 and params == 2:
        proba = np.arange(0.2, 1, .8/states)
        proba_neg = 1 - proba
        proba = np.array([proba, proba_neg]).T
        proba = np.expand_dims(proba, 0)
        likelihood_matrix = np.repeat(proba, agents, 0)
    elif strategy == 5 and params == 2:
        # min KL(theta_star||theta) grows
        probabilities_true = np.ones((agents, params))
        p_true = np.arange(0.35, 0.5, (0.5 - 0.35)/(agents-3))
        p_true = np.concatenate(([0.1, 0.1, 0.1], p_true))
        probabilities_true[:, 0] = p_true
        probabilities_true[:, 1] = 1 - p_true

        probabilities_other = np.ones((agents, params))
        probabilities_other[:, 0] = 1 - p_true
        probabilities_other[:, 1] = p_true

        if state_true is not None:
            likelihood_matrix = np.repeat(probabilities_other[:, np.newaxis, :], repeats=states, axis=1)
            for (agent, state) in zip(list(range(agents)), state_true):
                likelihood_matrix[agent, state, :] = probabilities_true[agent].reshape(-1)
        else:
            probabilities_true = probabilities_true[:, np.newaxis, :]
            probabilities_other = probabilities_other[:, np.newaxis, :]
            probabilities_other = np.repeat(probabilities_other, repeats=states - 1, axis=1)
            likelihood_matrix = np.concatenate((probabilities_true, probabilities_other), axis=1)
    else:
        raise ValueError("Invalid selection of the strategy.")
    return likelihood_matrix


class Generator():
    def __init__(self, likelihood_matrix, state_true, strategy):
        """
        :np.array likelihood_matrix: likelihood matrix of size agents x states x params
        :int state_true: true distributions state
        :int strategy:
            0, 1:discrete
            2:gaussian
        """
        self.likelihood_matrix = likelihood_matrix
        self.state_true = state_true
        if isinstance(self.state_true, int):
            self.state_true = self.state_true * np.ones(self.likelihood_matrix.shape[0])
            self.state_true = self.state_true.astype(int)
        self.strategy = strategy
        if strategy not in {0, 1, 2}:
            raise ValueError("Invalid selection of the strategy.")

    def sample(self):
        """
        :return: np.array(agents) sample
        """
        agents, states, params = self.likelihood_matrix.shape
        if (self.strategy == 0) or (self.strategy == 1):
            cumsum = np.cumsum(self.likelihood_matrix[np.arange(agents), self.state_true, :], axis=1)
            uniform = np.random.uniform(size=agents)
            sample = np.argmax(cumsum >= uniform[:, None], 1)
        elif self.strategy == 2:
            sample = np.random.normal(size=agents)
            mean = self.likelihood_matrix[np.arange(agents), self.state_true, 0]
            std = self.likelihood_matrix[np.arange(agents), self.state_true, 1]
            sample = mean * sample + std
        return sample


def kl_divergence(likelihood, state_0, state_1, option, state_true=None):
    """

    :param likelihood:
    :param state_0:
    :param state_1:
    :int option:
        0, 1:multinomial
        2: gaussian
    :return:
    """

    agents = likelihood.shape[0]
    if state_true is None:
        state_true = np.array([state_0]*agents)

    def helper_0(params_0, params_1):
        return np.sum(params_0 * np.log(params_0 / params_1))
    
    def helper_2(params_0, params_1):
        return params_1[1]/params_0[1] + (params_0[1]**2 + (params_0[0]-params_1[0])**2)/(2*params_1[1]**2)
        # log sigma2/sigma1 + (sigma1^2 + (mu1-mu2)^2)/(2sigma2^2)

    agents, _, _ = likelihood.shape
    # not efficient
    if option == 0 or 1:
        divergence = np.zeros(agents)
        likelihood_0 = likelihood[:, state_0, :]  # agents x parameters
        likelihood_1 = likelihood[:, state_1, :]
        likelihood_true = likelihood[np.arange(agents), state_true, :]
        # likelihood_true = np.zeros_like(likelihood_0)
        # for agent, state in zip(range(agents), state_true):
        #     likelihood_true[agent] = likelihood[agent, state, :]
        for agent in range(agents):
            divergence[agent] = helper_0(likelihood_true[agent, :], likelihood_1[agent, :]) \
                - helper_0(likelihood_true[agent, :], likelihood_0[agent, :])
    elif option == 2:
        divergence = np.zeros(agents)
        likelihood_0 = likelihood[:, state_0, :]  # agents x parameters
        likelihood_1 = likelihood[:, state_1, :]
        likelihood_true = likelihood[np.arange(agents), state_true, :]
        # likelihood_true = np.zeros_like(likelihood_0)
        # for agent, state in zip(range(agents), state_true):
        #     likelihood_true[agent] = likelihood[agent, state, :]
        for agent in range(agents):
            divergence[agent] = helper_2(likelihood_true[agent, :], likelihood_1[agent, :]) \
                - helper_2(likelihood_true[agent, :], likelihood_0[agent, :])
    else:
        raise ValueError("Invalid selection of the option.")
    return divergence

def random_combination_matrix_init(agents):
    matrix = np.random.uniform(size=(agents, agents))
    matrix = .5 * (matrix + matrix.T)
    matrix = matrix / matrix.sum(0)[None, :]
    return matrix


def estimation_error(matrix_1, matrix_2):
    #n = matrix_1.shape[0]
    #return 1 / n ** 2 * np.linalg.norm(matrix_1 - matrix_2)**2
    return np.linalg.norm(matrix_1 - matrix_2)**2


def estimate_adjacency(combination_matrix_learn):
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(combination_matrix_learn.reshape(-1, 1)).reshape(combination_matrix_learn.shape)
    if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
        clusters[clusters==0] = 2
        clusters[clusters==1] = 0
        clusters[clusters==2] = 1
    np.fill_diagonal(clusters, 1.)
    return clusters


def estimate_adjacency_colwise(combination_matrix_learn):
    n = combination_matrix_learn.shape[0]
    kmeans = KMeans(n_clusters=2)
    clusters = np.zeros_like(combination_matrix_learn)
    for col in range(n):
        clusters_col = kmeans.fit_predict(combination_matrix_learn[:, col].reshape(-1, 1)).reshape(-1)
        if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
            clusters_col[clusters_col==0] = 2
            clusters_col[clusters_col==1] = 0
            clusters_col[clusters_col==2] = 1
        clusters[:, col] = clusters_col
    # sym
    clusters = np.triu(clusters, 1) + np.triu(clusters, 1).T
    np.fill_diagonal(clusters, 1.)
    return clusters


def state_estimate(intermediate_belief, combination_matrix):
    def belief_estimate(intermediate_belief, combination_matrix):
        states, agents = intermediate_belief.shape
        denominator = np.zeros(agents)
        belief = np.zeros((states, agents))
        for state in range(states):
            belief[state] = np.exp(combination_matrix.T @ np.log(intermediate_belief[state, :]))
            denominator += belief[state]
        belief /= denominator[None, :]
        return belief

    belief = belief_estimate(intermediate_belief, combination_matrix)
    belief_state = np.argmax(belief, 0)
    values, counts = np.unique(belief_state, return_counts=True)
    state = values[np.argmax(counts)]
    return state


def state_estimate_psi(intermediate_belief, multitask=True):
    belief_state = np.argmax(intermediate_belief, 0)
    if not multitask:
        values, counts = np.unique(belief_state, return_counts=True)
        state = values[np.argmax(counts)]
    else:
        state = belief_state
    return state


def state_estimate_lk(expected_likelihood, state_0):
    if state_0 != 0:
        raise ValueError("state_0 = 0.")
    agents, states = expected_likelihood.shape
    states += 1

    expected_likelihood_expanded = np.zeros((agents, states, states))
    for state_nom in range(states):
        for state_denom in range(states):
            if state_nom == state_denom and state_nom == state_0:
                expected_likelihood_expanded[:, state_nom, state_denom] = 0
            elif state_nom == state_0:
                expected_likelihood_expanded[:, state_nom, state_denom] = expected_likelihood[:, state_denom - 1]
            elif state_denom == state_0:
                expected_likelihood_expanded[:, state_nom, state_denom] = - expected_likelihood[:, state_nom - 1]
            else:
                expected_likelihood_expanded[:, state_nom, state_denom] = expected_likelihood[:, state_denom - 1] \
                                                                          - expected_likelihood[:, state_nom - 1]
    argmax = expected_likelihood_expanded
    argmax[expected_likelihood_expanded > 0] = 1
    argmax[expected_likelihood_expanded <= 0] = 0
    argmax = argmax.sum(2)
    argmax = np.argmax(argmax, 1)
    return argmax


def state_estimate_psi_agent(intermediate_belief, agent):
    return np.argmax(intermediate_belief, 0)[agent]


def get_convex_and_lipschitz_constants(lambda_lim, step_size):
    # R is not correct
    R = lambda_lim.reshape(-1, 1) @ lambda_lim.reshape(1, -1)
    evalues = np.linalg.eigvals(R).real
    evalues[evalues<1e-10] = 0
    nu = (1-step_size)*(1-step_size)*min(evalues)
    theta = (1-step_size)*(1-step_size)*max(evalues)
    return nu, theta


def get_noise_norm_mean_and_std(lambda_lim, step_size):
    # R is not correct
    R = lambda_lim.reshape(-1, 1) @ lambda_lim.reshape(1, -1)
    evalues = np.linalg.eigvals(R).real
    evalues[evalues<1e-10] = 0
    beta = (1-step_size)*(1-step_size)*np.linalg.norm(R, 2)
    #sigma = 2*max(evalues)*(np.diag().sum())
    return beta#, sigma


def get_new_state(state_true, states):
    new_state = state_true + 1
    if new_state > states - 1:
        new_state = 0
    return new_state


def get_influences(centrality, log_lh, state_true):
    if state_true == 0:
        kl_infl = log_lh
    else:
        kl_infl = log_lh - log_lh[:, state_true-1][:, np.newaxis]
        kl_infl[:, 0] = - log_lh[:, state_true-1]
    kl_infl = kl_infl.sum(1) #agents
    return kl_infl * centrality