from pgmpy.estimators.StructureScore import StructureScore
import numpy as np
import pandas as pd
from math import log
from typing import Any, Dict, List, Optional

class BICScore_pgmpy(StructureScore):
    """
    Score learned by LLM, only for HC
    """
    def __init__(self, data, data_type, **kwargs):
        """
        data: learned for the CSL alg. a m * n numpy
        score_path: the path of LLM generated code.
        index: which alg is selected in the score_path_file. an int
        mode: discrete or continuous.
        """
        super(BICScore_pgmpy, self).__init__(data, **kwargs)
        self.data_type = data_type
        self.dataset = data.to_numpy()

    def local_score(self, variable, parents):

        score = local_score_BIC_Std(self.dataset, variable, parents, parameters={'data_type': self.data_type})

        return score

def local_score_BIC_Std_rev(Data: np.ndarray, child: int, parents: list[int], parameters : Dict[str, Any]) ->float:

    score = local_score_BIC_Std(Data, child, parents, parameters )
    return -1 * score

def local_score_BIC_Std(Data: np.ndarray, child: int, parents: list[int], parameters : Dict[str, Any]) -> float:
    
    data_type = parameters['data_type']
    if data_type == 'discrete':

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
    
    elif data_type == 'continuous':

        n = Data.shape[0]
        lambda_value = 2

        cov = np.cov(Data.T)
            
        if len(parents) == 0:
            H = np.log(cov[child, child])
            score = -1 * (n * H)
            score = score.item()
            return score

        yX = cov[np.ix_([child], parents)]
        XX = cov[np.ix_(parents, parents)]
        H = np.log(cov[child, child] - yX @ np.linalg.inv(XX) @ yX.T)

        score = -1 * (n * H + np.log(n) * len(parents) * lambda_value * 0.5)

        score = score.item()
        return score