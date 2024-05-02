import numpy as np


def kl_divergence_estimation_(beliefs, combination_matrix, step_size, window):
    states, agents = beliefs[0].shape
    kl_div = np.zeros((agents, states - 1))
    log_beliefs = []
    for time in range(len(beliefs)):
        log_beliefs.append(np.array([np.log(
            beliefs[time][0, :] / \
            beliefs[time][n, :]) for n in range(1, states)
        ]).T)
    for log_prev, log in zip(log_beliefs[-window:-1], log_beliefs[-window+1:]):
        kl_div += log - (1-step_size) * combination_matrix.T @ log_prev
    kl_div /= step_size * window
    return kl_div


def kl_divergence_estimation(beliefs, combination_matrix, step_size, window):
    states, agents = beliefs[0].shape
    kl_div = np.zeros((agents, states - 1))
    log_beliefs = []
    for time in range(window, 0, -1):
        log_beliefs.append(np.array([np.log(
            beliefs[-time-1][0, :] / \
            beliefs[-time-1][n, :]) for n in range(1, states)
        ]).T)
    for log_prev, log in zip(log_beliefs[:-1], log_beliefs[1:]):
        kl_div += log - (1-step_size) * combination_matrix.T @ log_prev
    kl_div /= step_size * window
    return kl_div


def optimization_step(log_cur, log_prev, adj_matrix_prev, kl_div, lr=0.05, alpha=0.,
                      log_prev_M=None, step_size=None, projection=True, multistate=False):
    if step_size is None:
        multiplier_1 = 1.
        multiplier_2 = 1.
    else:
        multiplier_1 = 1. - step_size
        multiplier_2 = step_size
    if not multistate:
        log_cur = log_cur.reshape([-1, 1])
        log_prev = log_prev.reshape([-1, 1])
        log_prev_M = log_prev.reshape([-1, 1])
        kl_div = kl_div.reshape([-1, 1])
    if log_prev_M is None:
        log_prev_M = log_prev
    adj = adj_matrix_prev.T + lr*(
            multiplier_1*(log_cur - multiplier_1*adj_matrix_prev.T@log_prev - multiplier_2*kl_div)@log_prev_M.T -\
            alpha*(adj_matrix_prev.T / np.abs(adj_matrix_prev.T))
    )
    adj = adj.T
    if projection:
        # positive
        adj[adj < 0.] = 0.
        # normalized
        adj = adj / adj.sum(0)[None, :]
    return adj


def get_loss(combination_matrix, log_prev, log_cur, kl_div, step_size=None, multistate=False):
    if step_size is None:
        multiplier_1 = 1.
        multiplier_2 = 1.
    else:
        multiplier_1 = 1. - step_size
        multiplier_2 = step_size
    if not multistate:
        log_cur = log_cur.reshape([-1, 1])
        log_prev = log_prev.reshape([-1, 1])
        kl_div = kl_div.reshape([-1, 1])
    return .5*np.linalg.norm(log_cur - multiplier_1*combination_matrix.T@log_prev - multiplier_2*kl_div, ord=2)**2