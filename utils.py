import numpy as np
import re
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.SHD import SHD
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from CausalModel.BICScore import local_score_BIC_Std
from pgmpy.base.DAG import DAG
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.PDAG2DAG import pdag2dag

def GetGraphBICScore(dag, dataset, data_type):

    n = dag.shape[0]
    score_sum = 0
    for child in range(n):
        parents = np.where(dag[:,child] != 0)[0]
        score_sum += local_score_BIC_Std(Data=dataset, child=child,parents=parents,parameters={'data_type': data_type})
    
    return score_sum

def ConvertBNtoDAG(bn:DAG):

    n = len(bn.nodes())
    dag = np.zeros((n,n))
    for edge in bn.edges():
        dag[edge[0], edge[1]] = 1
    return dag

def ConvertBNtoDAG_OnlyForHC_Continuous(bn:DAG):
    
    n = len(bn.nodes())
    Dag = np.zeros((n,n))
    for edge in bn.edges():
        from_index = int(edge[0][1])
        to_index = int(edge[1][1])
        Dag[from_index, to_index] = 1
    return Dag
    
def ConvertGeneralGraphtoDAG(G: GeneralGraph):

    ## if contain undirected edges:
    if np.isin(-2, G.graph + G.graph.T):
        G_new = pdag2dag(G)
        std = G_new.graph
    else:
        std = G.graph
    
    return (-1 * std + np.abs(std)) * 0.5


def compute_SHD(learned:np.ndarray, std:np.ndarray):

    n = std.shape[0]

    name_vars = [f'X{i}' for i in range(1, n + 1)]
    nodes = []
    for name in name_vars:
        node = GraphNode(name)
        nodes.append(node)
    G_learned = GeneralGraph(nodes)
    G_std = GeneralGraph(nodes)

    G_learned.graph = -1 * learned + learned.T
    G_learned.reconstitute_dpath(G_learned.get_graph_edges())

    G_std.graph = -1 * std + std.T
    G_std.reconstitute_dpath(G_std.get_graph_edges())
    
    shd = SHD(truth=dag2cpdag(G_std), est=dag2cpdag(G_learned)).get_shd()

    return shd

def compute_NHD(learned_dag:np.ndarray, std_dag: np.ndarray):

    n = learned_dag.shape[0]
    NHD = 1/(n*n) * np.sum(np.abs(std_dag - learned_dag))

    return NHD
    
def IsDag(graph):

    n = len(graph)

    visited = [0] * n
    
    def dfs(node):

        if visited[node] == 1:
            return False

        if visited[node] == 2:
            return True

        visited[node] = 1
        for neighbor in range(n):
            if graph[node][neighbor] == 1:
                if not dfs(neighbor):
                    return False
        
        visited[node] = 2
        return True
    

    for i in range(n):
        if visited[i] == 0:
            if not dfs(i):  
                return False
    
    return True 

def compute_struturalF1(learned:np.ndarray, std:np.ndarray):

    n = std.shape[0]

    name_vars = [f'X{i}' for i in range(1, n + 1)]
    nodes = []
    for name in name_vars:
        node = GraphNode(name)
        nodes.append(node)
    G_learned = GeneralGraph(nodes)
    G_std = GeneralGraph(nodes)

    G_learned.graph = -1 * learned + learned.T
    G_learned.reconstitute_dpath(G_learned.get_graph_edges())

    G_std.graph = -1 * std + std.T
    G_std.reconstitute_dpath(G_std.get_graph_edges())
    
    ### For arrows:
    arrow = ArrowConfusion(G_std, G_learned)
    arrowsTp = arrow.get_arrows_tp()
    arrowsFp = arrow.get_arrows_fp()
    arrowsFn = arrow.get_arrows_fn()
    arrowsTn = arrow.get_arrows_tn()

    arrowPrec = arrow.get_arrows_precision()
    arrowRec = arrow.get_arrows_recall()

    if arrowPrec + arrowRec == 0:
        arrowF1 = 0
    else:
        arrowF1 = 2*arrowPrec*arrowRec/(arrowPrec + arrowRec)

    ### For adjacents:
    adj = AdjacencyConfusion(G_std, G_learned)

    adjTp = adj.get_adj_tp()
    adjFp = adj.get_adj_fp()
    adjFn = adj.get_adj_fn()
    adjTn = adj.get_adj_tn()

    if adjTp + adjFp == 0:
        adjPrec = 0
    else:
        adjPrec = adj.get_adj_precision()

    if adjTp + adjFn == 0:
        adjRec = 0
    else:
        adjRec = adj.get_adj_recall()


    adjRec = adj.get_adj_recall()

    if adjRec + adjPrec == 0:
        adjF1 = 0
    else:
        adjF1 = 2 * adjPrec * adjRec / (adjRec + adjPrec)

    results = {'arrowPrec': round(arrowPrec, 4), 'arrowRec': round(arrowRec,4), 'arrowF1': round(arrowF1,4),
               'adjPrec': round(adjPrec,4), 'adjRec': round(adjRec,4), 'adjF1': round(adjF1,4)}
    return results


def get_real():

    from pathlib import Path
    import pandas as pd
    import numpy as np
    current_file_path = Path(__file__).resolve()
    parent_folder = current_file_path.parent
    dataset = f'{parent_folder}\\Real_data\\COVID-19_real_discrete_quartiles.csv'
    dag = f'{parent_folder}\\Real_data\\DAGtrue_COVID-19.csv'
    df1 = pd.read_csv(dag)

    unique_vars = pd.concat([df1['Var 1'], df1['Var 2']]).unique()
    var_mapping = {var: idx for idx, var in enumerate(unique_vars)}

    adj_matrix = np.zeros((len(var_mapping), len(var_mapping)), dtype=int)

    for _, row in df1.iterrows():
        var1_idx = var_mapping[row['Var 1']]
        var2_idx = var_mapping[row['Var 2']]
        adj_matrix[var1_idx, var2_idx] = 1  

    df2 = pd.read_csv(dataset)

    df2_var_names = df2.columns.to_list()
    var_name_to_int = {var: var_mapping[var] for var in df2_var_names}


    sorted_columns = sorted(df2.columns, key=lambda x: var_name_to_int[x])
    df2 = df2[sorted_columns]


    df2.columns = [f'X{i+1}' for i in range(df2.shape[1])]

    for column in df2.columns:
        df2[column] = df2[column].astype('category').cat.codes
    
    data = df2.to_numpy()

    return adj_matrix,data


