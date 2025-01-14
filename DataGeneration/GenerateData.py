import os
import jpype
import jpype.imports
from pathlib import Path
current_directory = Path.cwd()
folder_C = current_directory / 'pytetrad/tetrad-current.jar'
# jpype.startJVM(classpath=[folder_C])
jpype.startJVM(
    jpype.getDefaultJVMPath(),
    '-Xmx8g', 
    '-Djava.class.path={}'.format(folder_C)
)
import edu.cmu.tetrad.data as data
from edu.cmu.tetrad.util import Params, Parameters
from edu.cmu.tetrad.algcomparison.simulation import GeneralSemSimulation, BayesNetSimulation, LeeHastieSimulation
import edu.cmu.tetrad.algcomparison.graph.RandomForward as RandomForward
import edu.cmu.tetrad.algcomparison.graph.ScaleFree as ScaleFree
import edu.cmu.tetrad.algcomparison.graph.ErdosRenyi as ErdosRenyi

import numpy as np
import pandas as pd
import argparse
import random

def generate_data(opts):
    '''
    cited from https://github.com/Gerlise/AutoCD/blob/main/autocd/generate_data.py
    Generate graphical model and simulate data.

    :param opts: information to generate graphical models and simulate data
    :param datatype: the data type of the dataset
    :param seed: random seed
    :return: Simulated dataset and the generated DAG.
    '''

    min_c = 2
    max_c = random.randint(3,5)
    if opts.nodes >= 40: 
        max_c = 3
    perc_d = 50
    datatype = opts.data_type[0]


    params = Parameters()
    if opts.graph_type[0] == 'randomgraph':
        random_graph = RandomForward()  # Erdos-Renyi, Scale-free, ...
    elif opts.graph_type[0] == 'erdosrenyi':
        random_graph = ErdosRenyi()  # Erdos-Renyi, Scale-free, ...
        params.set(Params.PROBABILITY_OF_EDGE, 0.2)
    elif opts.graph_type[0] == 'scalefree':
        random_graph = ScaleFree()  # Erdos-Renyi, Scale-free, ...
    params.set(Params.NUM_MEASURES, opts.nodes)
    params.set(Params.SAMPLE_SIZE, opts.instances)
    params.set(Params.AVG_DEGREE, opts.degree)
    params.set(Params.MAX_DEGREE, opts.degree + 1)

    params.set(Params.DIFFERENT_GRAPHS, True)
    params.set(Params.NUM_LATENTS, 0)  # Can be changed, it is not considered in this version
    params.set(Params.RANDOMIZE_COLUMNS, False)
    params.set(Params.SEED, opts.seed)

    if datatype == "continuous" or 'discrete':
        params.set(Params.MIN_CATEGORIES, min_c)
        params.set(Params.MAX_CATEGORIES, max_c)
        sim = BayesNetSimulation(random_graph)
    elif datatype == "mixed":
        params.set(Params.MIN_CATEGORIES, min_c)
        params.set(Params.MAX_CATEGORIES, max_c)
        params.set(Params.PERCENT_DISCRETE, perc_d)
        sim = LeeHastieSimulation(random_graph)
    else:
        print("This data type does not exist")

    sim.createData(params, True)
    t_dag = sim.getTrueGraph(0)
    t_data = sim.getDataModel(0)

    # Save data and target graph
    if datatype == 'continuous':
        dag = convert_tetrad_graph(t_dag)
        data = add_data_for_sem(dag, opts.noise_type[0], opts.instances)
        data = data.to_numpy().astype(np.float32)
    else:
        data = convert_tetrad_data(t_data, datatype)
        dag = convert_tetrad_graph(t_dag)
        data = data.to_numpy().astype(np.int8)
    return data, dag

def add_data_for_sem(dag, error_type, m_samples):

    n = dag.shape[0]

    data = np.zeros((m_samples,n))
    for child in range(n):
        delta = random.uniform(1,2)
        if error_type == 'gaussian':
            data[:,child] = np.random.normal(loc=0, scale = delta, size=m_samples)
        elif error_type == 'gumbel':
            data[:,child] = np.random.gumbel(loc=0, scale = delta, size=m_samples)
        elif error_type == 'exp':
            data[:,child] = np.random.exponential(scale = delta, size=m_samples)
        else:
            TypeError('wrong noise type!')
        parents = np.where(dag[:, child] != 0)[0]
        for parent in parents:
            beta = random.uniform(-1,1)
            new_data = data[:,child] +  beta * data[:,parent]
            data[:,child] = new_data
    
    dataset = pd.DataFrame(data)
            
    return dataset

def convert_tetrad_graph(graph):
    '''
    cited from https://github.com/Gerlise/AutoCD/blob/main/autocd/generate_data.py
    Create adjacency matrix from Tetrad DAG.
    :param graph: Tetrad Graph object - directed acyclic graph (DAG)
    :return: Adjacency matrix of the given graph.
    '''

    dag_map = {"ARROW": 1, "TAIL": 0, "NULL": 0, "CIRCLE": 0}
    n_nodes = graph.getNumNodes()
    nodes = graph.getNodes()
    matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    for edge in graph.getEdges():
        i = nodes.indexOf(edge.getNode1())
        j = nodes.indexOf(edge.getNode2())

        matrix[i, j] = dag_map[edge.getEndpoint2().name()]
        matrix[j, i] = dag_map[edge.getEndpoint1().name()]

    return matrix


def convert_tetrad_data(data, datatype):
    '''
    cited from https://github.com/Gerlise/AutoCD/blob/main/autocd/generate_data.py
    Create pandas dataframe from Tetrad data object.
    :param data: Tetrad DataModel object - dataset
    :return: Dataframe of the dataset.
    '''

    names = data.getVariableNames()
    columns_ = []

    for name in names:
        columns_.append(str(name))

    df = pd.DataFrame(columns=columns_, index=range(data.getNumRows()))

    for row in range(data.getNumRows()):
        for col in range(data.getNumColumns()):
            df.at[row, columns_[col]] = str(data.getObject(row, col))

    if datatype == "continuous":
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype(int)
    elif datatype == "discrete":
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype(float)

    return df

def parse_args(args):
    '''
    set the params in data generation
    :return:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nodes", type=int, default=10, help="Number of nodes in the DAG (default 10)"
    )
    parser.add_argument(
        "--degree", type=int, default=3, help="Average node degree in the DAG (default 3)"
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=1000,
        help="Number of instances in the dataset (default is 1000)",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        nargs="+",
        choices=["continuous", "discrete", "mixed"],
        default="discrete",
        help="Data type of the graph models: continuous, discrete and/or mixed",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of repetitions of the dataset (default is 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed"
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        nargs="+",
        choices=["randomgraph", "erdosrenyi", "scalefree"],
        default="randomgraph",
        help="Type of generated graph",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        nargs="+",
        choices=["gaussian", "gumbel", "exp"],
        default="gaussian",
        help="Type of noise",
    )
    opts = parser.parse_args(args)
    return opts


