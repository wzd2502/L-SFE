from pgmpy.estimators.StructureScore import StructureScore
import numpy as np
import pandas as pd
import json
from math import log
from typing import Any, Dict, List, Optional

class LLMScore_pgmpy(StructureScore):
    """
    Score learned by LLM, only for HC
    """
    def __init__(self, data, score_path, **kwargs):
        """
        data: learned for the CSL alg. a m * n numpy
        score_path: the path of LLM generated code.
        index: which alg is selected in the score_path_file. an int
        """
        super(LLMScore_pgmpy, self).__init__(data, **kwargs)

        ### the data used in llm need a np.array input.
        self.dataset = data.to_numpy()

        self.select_alg = score_path

    def local_score(self, variable, parents):
        
        # try:
        exec(self.select_alg, globals())
        """
        funcname: subscoreLLM(child, parents, Data)
        """
        score = subscoreLLM(variable, parents, self.dataset)

        if isinstance(score, np.ndarray):
            score =score.item()
        # except:
        #     RuntimeError('LLM generated code cannot run!')

        return score

### For causal-learn package.
def llmscore(Data: np.ndarray, variable: int, parents: list[int], parameters: Dict[str, Any]) ->float:

    code = parameters['code']
    # try:
    exec(code, globals())
    score = subscoreLLM(variable, parents, Data)
    # except:
    #     RuntimeError('LLM generated code cannot run!')

    return -score
