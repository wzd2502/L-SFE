import numpy as np
import pandas as pd
from .GenerateData import parse_args,generate_data
import random

class training:

    def __init__(self):
        self.n_vars = []
        self.m_samples = []
        self.dataset = []
        self.stdDAG = []

        self.Score_loss = []
        self.SHD_loss = []
        self.NHD_loss = []

        self.data_type = ''
        self.noise_type = ''
        self.graph_type = ''
        self.train_method = ''

def generate_train(n_vars:list, m_samples:list, **kwargs):
    '''

    :param times: times
    :return: generate a series of tranining settings.
    '''
    graph_type = kwargs.get('graph_type', 'randomgraph')
    data_type = kwargs.get('data_type', 'discrete')
    noise_type = kwargs.get('noise_type', 'gaussian')
    seed = kwargs.get('seed', 2512)
    train_method = kwargs.get('train_method', 'HC')
    training_tool = training()
    for i in range(len(n_vars)):
        for j in range(len(m_samples)):
            if noise_type == 'both':
                noise_type_sub = random.choice(['exp', 'gumbel'])
            else:
                noise_type_sub = noise_type
            if data_type == 'discrete':
                degree_sub = str(random.randint(2,6))
            else:
                degree_sub = '4'
            opts = parse_args(['--nodes', str(n_vars[i]), '--degree', degree_sub, '--instances', str(m_samples[j]), 
                               '--seed', str(seed), '--graph_type', graph_type, '--data_type', data_type, '--noise_type', noise_type_sub])
            dataset, dag = generate_data(opts)

            # dismiss the influence of ordering:
            col_indices = np.random.permutation(dataset.shape[1]) 
            dataset = dataset[:, col_indices]
            dag = dag[col_indices, :][:, col_indices]

            training_tool.dataset.append(dataset)
            training_tool.stdDAG.append(dag)
            training_tool.n_vars.append(i)
            training_tool.m_samples.append(j)

    training_tool.data_type = data_type
    training_tool.graph_type = graph_type
    training_tool.noise_type = noise_type
    training_tool.train_method = train_method
    return training_tool