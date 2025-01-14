import multiprocessing.queues
import multiprocessing
import numpy as np
import pandas as pd
import re
from DataGeneration.TrainDataSet import training
from utils import IsDag, compute_SHD, compute_NHD
from CausalModel.LLMHC import hc_llm
from CausalModel.LLMGES import ges_llm
from CausalModel.LLMBOSS import boss_llm
from utils import GetGraphBICScore, ConvertBNtoDAG, ConvertGeneralGraphtoDAG
from alive_progress import alive_bar

def wrapper_Loss(code, dataset, method, result_queue):

    try:
        if method == 'HC':
            data = pd.DataFrame(dataset)
            est = hc_llm(data=data)
            bn = est.estimate(scoring_method='llmscore', show_progress= False, llm_path=code)
            dag = ConvertBNtoDAG(bn)
        elif method == 'GES':
            cm = ges_llm(X=dataset, score_func= 'llmscore', llm_path=code)
            dag = ConvertGeneralGraphtoDAG(cm['G'])
        elif method == 'BOSS':
            cm = boss_llm(X=dataset, score_func= 'llmscore', llm_path=code)
            dag = ConvertGeneralGraphtoDAG(cm)
        else:
            ValueError('No such training method!')
        result_queue.put(dag)
    except:
        print('Failed running!')
        n = dataset.shape[1]
        dag = np.zeros((n,n), dtype=np.int8)
        result_queue.put(dag)

class Alg:
    def __init__(self, source):
        '''

        :param type: the type of heuristic, one of ['Directed Graph', 'Completely Partially Directed Acyclic Graph', 'Topological Ordering']
        :param code: the implementation of this heuristic, a json file.
        :param loss: the loss of this heuristic. float.
        '''
        self.source = source
        self.code = None
        self.loss = None
        self.shd = []
        self.score = []
        self.success = False

    def todict(self):

        return {
            'source': self.source,
            'code': self.code,
            'loss': str(self.loss),
            'shd': str(self.shd),
            'score': str(self.score)
        }
    def GetCode(self, inicode):
        
        # If LLM redefine these two function, discard it.
        if "def IsDag" in inicode or "def get_deltascore" in inicode:
            return False

        pattern = r"python(.*?)```"
        match = re.search(pattern, inicode, re.DOTALL)

        if match:
            extracted_content = match.group(1).strip()
        else:
            ### only for seeds
            extracted_content = re.findall(r"import.*", inicode, re.DOTALL)[0]
        self.code = extracted_content

        return True

    def GetLoss(self, training_tool:training):

        if self.code is None:
            raise Exception('code is not found!')

        results = self.LLM_SS(training_tool)

        self.loss = results['avg_nhd']
        self.shd = results['shd']
        self.score = results['score']
        self.success = results['success']

    
    def LLM_SS(self, training_tool:training):
        
        shdcollection = []
        scorecollection = []
        sum_nhd = 0
        jump = False

        datasets = training_tool.dataset
        stdDAGs = training_tool.stdDAG
        method = training_tool.train_method
        data_type = training_tool.data_type
        num_task = len(datasets)

        with alive_bar(num_task) as bar:
            for dataset, stdDAG in zip(datasets, stdDAGs):
                
                n = dataset.shape[1]
                m = dataset.shape[0]
                if not jump:

                    result_queue = multiprocessing.Queue()
                    process = multiprocessing.Process(target=wrapper_Loss, args=(self.code, dataset, method,  result_queue))

                    process.start()
                    process.join(timeout=300)

                    if process.is_alive():
                        print(f"Process {process.name} is still running after {300} seconds, terminating it.")
                        process.terminate()
                        process.join()
                        new_dag = np.zeros((n, n), dtype=np.int8)
                        jump = True
                    else:
                        new_dag = result_queue.get()
                        if np.sum(new_dag) == 0:
                            # print('searched for empty graph!')
                            jump = True

                    if not IsDag(new_dag):
                        # print('results contain cycles!')
                        new_dag = np.zeros((n, n), dtype=np.int8)
                        jump = True
                else:
                    new_dag = np.zeros((n, n), dtype=np.int8)

                shd = compute_SHD(learned= new_dag, std= stdDAG)
                new_score = GetGraphBICScore(dag= new_dag, dataset= dataset, data_type= data_type)
                nhd = compute_NHD(learned_dag= new_dag, std_dag= stdDAG)

                # print(f'Network: {n} vars, {m} samples. Whether Alg jumps: {jump}. Score: {new_score}, shd: {shd}. nhd: {nhd}')
                shdcollection.append(shd)
                scorecollection.append(new_score)
                sum_nhd += nhd

                bar()

        avg_nhd = sum_nhd/len(datasets)
        return {'shd': shdcollection, 'score': scorecollection, 'avg_nhd': avg_nhd, 'success': not jump}