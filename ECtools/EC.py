from ECtools.heuristic import Alg
from LLMtools.LossLLM import LossLLM
import random
from DataGeneration.TrainDataSet import training
from ECtools.seeds import seedpool
import json
import numpy as np
import time
import os


class ECforAlg:

    def __init__(self, model:LossLLM, trainingtool:training):
        self.n_population = 1
        self.max_iterations = 5
        self.n_in_population = 5
        self.crossover_rate = 0.2
        self.mutation_rate = 0.2
        self.newadd_rate = 0.2
        self.algs = list()

        self.seed_algs = seedpool()
        self.num_seed_alg = 1

        if model is None:
            raise Exception('LLM object has not been initialized')
        else:
            self.LLM = model

        if trainingtool is None:
            raise Exception('Training tool has not been initialized')
        else:
            self.training_tool = trainingtool

        if trainingtool.data_type == 'discrete':
            self.file = 'Generated_Code_Discrete'
            self.seed_alg  = self.seed_algs['discrete']
        else:
            self.file = 'Generated_Code_Continuous'
            self.seed_alg  = self.seed_algs['continuous']

    def Inital(self, para):

        while True:
            start_time = time.time()
            # if para < 0:
            if para < self.num_seed_alg:
                newalg = Alg(source = 'seed')
                code = self.seed_alg
            elif para < self.n_in_population * self.n_population:
                newalg = Alg(source =  'initial')
                code = self.LLM.Initilization()
            else:
                raise Exception('Invalid parameter for initialization')
            if newalg.GetCode(code):
                newalg.GetLoss(self.training_tool)
            else:
                continue

            end_time = time.time()
            execution_time = end_time - start_time
            if newalg.success:
                break
        
        print(f'{para} -th initial complete! The loss of new Alg is: {str(newalg.loss)}, the shd of new Alg is {str(newalg.shd)}, the score of new Alg is {str(newalg.score)}, the time cost of new Alg is {execution_time:.4f}s')
        return newalg

    def Evolution(self, para, pool):
        
        while True:
            start_time = time.time()
            if para == 'crossover':
                alg_temp = random.sample(pool, 2)
                code= self.LLM.CrossOver(alg_temp[0], alg_temp[1])
                newalg = Alg('crossover')
        
            elif para == 'mutation':

                alg_temp = random.sample(pool, 1)
                if np.random.random() < 0.5:
                    code= self.LLM.Mutation1(alg_temp[0])
                else:
                    code= self.LLM.Mutation2(alg_temp[0])
                newalg = Alg('mutation')

            elif para == 'newadd':
                newalg = Alg('newadd')
                code= self.LLM.NewAdd(pool)
            else:
                raise Exception('Invalid parameter for evolution')
            
            if newalg.GetCode(code):
                newalg.GetLoss(self.training_tool)
            else:
                continue

            end_time = time.time()
            execution_time = end_time - start_time
            if newalg.success:
                break

        print(f'A new {para} complete! The loss of new Alg is: {str(newalg.loss)}, the shd of new Alg is {str(newalg.shd)}, the score of new Alg is {str(newalg.score)}, the time cost of new Alg is {execution_time:.4f}s')
        return newalg

    def PopBestAlg(self):

        self.algs = sorted(self.algs, key=lambda obj: obj.loss, reverse=False)

        print(f'For the LLM desiged alg, shd: {self.algs[0].shd}, score: {self.algs[0].score},\n avg_NHD: {self.algs[0].loss}')

        results = {'method': 'LLM', 'score': self.algs[0].score, 'shd': self.algs[0].shd, 'avg_NHD': self.algs[0].loss}

        return results

    def writetojson(self, name, mode):

        folder = os.path.dirname(name)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if mode == 'All':
            data_to_write = [alg.todict() for alg in self.algs]
        elif mode == 'Best':
            data_to_write = [self.algs[0].todict()]

        with open(name, 'w') as json_file:
            json.dump(data_to_write, json_file, indent=4)

    def run_Inital(self):

        if self.LLM is None:
            raise Exception('LLM not initialized')

        if self.training_tool is None:
            raise Exception('training_tool not initialized')

        ### future works.

        print('--------------Initialization Start--------------')
        paras = list(range(0, self.n_in_population))

        results = []
        for para in paras:
            
            results.append(self.Inital(para))

        self.algs = self.algs + results

        ## If the best dags need to be updated
        # self.update_dag_pool()

        self.algs = sorted(self.algs, key=lambda obj: obj.loss, reverse=False)[:self.n_population * self.n_in_population]

        self.writetojson(name=f'{self.file}/All/Initialization.json', mode='All')

        self.writetojson(name=f'{self.file}/Best/Initialization.json', mode='Best')


    def run_Evolution(self):

        if len(self.algs) < self.n_population * self.n_in_population:
            raise Exception('Invalid Initialization')

        num_crossover_alg = int(len(self.algs) * self.crossover_rate)
        num_mutation_alg = int(len(self.algs) * self.mutation_rate)
        num_newadd_alg = int(len(self.algs) * self.newadd_rate)

        for i in range(self.max_iterations):
            temp_algs = list()
            print(f'--------------{i}-th Evolution Start--------------')

            results = []
            for j in range(num_newadd_alg):

                results.append(self.Evolution('newadd', self.algs))
            
            self.algs = self.algs + results

            results = []

            for k in range(num_crossover_alg):

                results.append(self.Evolution('crossover', self.algs))
            
            for l in range(num_mutation_alg):

                results.append(self.Evolution('mutation', self.algs))

            self.algs = self.algs + results

            self.algs = sorted(self.algs, key=lambda obj: obj.loss, reverse=False)[:self.n_population * self.n_in_population]

            print(f'{i} -th evolution complete!, The loss of best Alg is: {self.algs[0].loss}')

            self.writetojson(name = f'{self.file}/All/{i}-th evolution.json', mode = 'All')

            self.writetojson(name = f'{self.file}/Best/{i}-th evolution.json', mode = 'Best')












