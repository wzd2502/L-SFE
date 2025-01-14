from pgmpy.estimators.StructureScore import StructureScore
from causallearn.score.LocalScoreFunction import local_score_cv_general,local_score_marginal_general

class cv_general(StructureScore):
    """
    Score learned by LLM, only for HC
    """
    def __init__(self, data, **kwargs):
        """
        data: learned for the CSL alg. a m * n numpy
        score_path: the path of LLM generated code.
        index: which alg is selected in the score_path_file. an int
        mode: discrete or continuous.
        """
        super(cv_general, self).__init__(data, **kwargs)
        self.dataset = data.to_numpy()
        self.parameters = {
                "kfold": 10,  # 10 fold cross validation
                "lambda": 0.01,
            }  # regularization parameter

    def local_score(self, variable, parents):

        score = local_score_cv_general(Data = self.dataset, Xi= variable, PAi= parents, parameters= self.parameters)
        
        return score
    
class marginal_general(StructureScore):
    """
    Score learned by LLM, only for HC
    """
    def __init__(self, data, **kwargs):
        """
        data: learned for the CSL alg. a m * n numpy
        score_path: the path of LLM generated code.
        index: which alg is selected in the score_path_file. an int
        mode: discrete or continuous.
        """
        super(marginal_general, self).__init__(data, **kwargs)
        self.dataset = data.to_numpy()
        self.parameters = {}


    def local_score(self, variable, parents):

        score = local_score_marginal_general(Data = self.dataset, Xi= variable, PAi= parents, parameters= self.parameters)
        
        return score
