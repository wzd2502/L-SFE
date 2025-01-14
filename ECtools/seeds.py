def seedpool():

    algs = dict()

    alg19 = """
import numpy as np
def subscoreLLM(child, parents, Data):
    # child: an index;
    # parents: a list of index, it is [] when child is a root node;
    # Data: a m*n numpy, where m is the number of instances, and n is the numer of variables.

    m = Data.shape[0]
    lambda_value = 2
    cov = np.cov(Data.T)
        
    if len(parents) == 0:
        H = np.log(cov[child, child])
        score = -1 * (m * H)
        return score

    yX = cov[np.ix_([child], parents)]
    XX = cov[np.ix_(parents, parents)]
    H = np.log(cov[child, child] - yX @ np.linalg.inv(XX) @ yX.T)

    score = -1 * (m * H + np.log(m) * len(parents) * lambda_value * 0.5)
    score = score.item()
    return score
"""

    alg20 = """
import numpy as np
import pandas as pd
def subscoreLLM(child, parents, Data):
    # child: an index;
    # parents: a list of index, it is [] when child is a root node;
    # Data: a m*n numpy, where m is the number of instances, and n is the numer of variables.

    lambda_value = 1
    var_states = np.unique(Data[:,child])
    var_cardinality = len(var_states)

    if len(parents) != 0:
        num_parents_states = np.prod([len(np.unique(Data[:,var])) for var in parents])
        state_counts = pd.crosstab(index=[Data[:,child]], columns=[Data[:,var] for var in parents])
    else:
        num_parents_states = 1
        _, state_counts = np.unique(Data[:, child], return_counts=True)
    sample_size = Data.shape[0]

    counts = np.asarray(state_counts)
    log_likelihoods = np.zeros_like(counts, dtype=float)

    # Compute the log-counts
    np.log(counts, out=log_likelihoods, where=counts > 0)

    # Compute the log-conditional sample size
    log_conditionals = np.sum(counts, axis=0, dtype=float)
    if isinstance(log_conditionals, np.ndarray):
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)
    else:
        log_conditionals = [np.log(log_conditionals)]

    # Compute the log-likelihoods
    log_likelihoods -= log_conditionals
    log_likelihoods *= counts

    score = np.sum(log_likelihoods)
    score -= 0.5 * lambda_value * log(sample_size) * num_parents_states * (var_cardinality - 1)

    return score
"""
    algs['discrete'] = alg20
    algs['continuous'] = alg19
    return algs


