import logging
logging.basicConfig(level=logging.ERROR)

from DataGeneration.TrainDataSet import generate_train
from ECtools.EC import ECforAlg
from LLMtools.LossLLM import LossLLM
import argparse


def main():

    parser = argparse.ArgumentParser(description= 'This function can help you to training your own score function or test its performance.')
    parser.add_argument('--n_vars', type=int, nargs= '+', required=True, default=10, help='The number of variables, support multi-inputs in one time, but use space to seperate.' )
    parser.add_argument('--m_samples', type=int, nargs='+', required=True, default=1000,  help= 'The number of samples, support multi-inputs in one time, but use space to seperate.')
    parser.add_argument('--data_type', type=str, choices=['discrete', 'continuous'], required=True, default='discrete', help= 'The data type of generated data.')
    parser.add_argument('--noise_type', type=str, choices=['exp', 'gaussian', 'gumbel', 'mixed'], required=False, default= 'gaussian', help='The noise type of data if it is generated from SEM.')
    parser.add_argument('--graph_type', type=str, choices=['randomgraph', 'erdosrenyi', 'scalefree'], default='randomgraph', required=True, help='The type of std graph.')
    parser.add_argument('--degree',type = int, required=False, default= 4, help='The avg degree of std graph.')
    parser.add_argument('--train_method', type = str, choices=['HC', 'GES', 'BOSS'], required=False, default='HC', help='Search Alg used in training.')
    parser.add_argument('--llm_type', type=str, default='gpt-4o-mini-2024-07-18', help='LLM used for training.')
    parser.add_argument('--llm_api', type=str,  help='Key api of used LLM.')
    parser.add_argument('--llm_url', type=str, help='website of the LLM.')
    parser.add_argument('--seed', type=int, required = False, default= 1234, help='The random seed')
    args = parser.parse_args()
    
    n_vars = args.n_vars
    n_samples = args.m_samples
    graph_type = args.graph_type
    data_type = args.data_type
    noise_type = args.noise_type
    seed = args.seed
    train_method = args.train_method
    llm_mode = args.llm_type

    print('----------Training Mode----------')
    training_tool = generate_train(n_vars, n_samples, graph_type = graph_type, data_type = data_type, 
                                noise_type = noise_type, seed = seed, train_method = train_method)
    
    print('Training settings finished.')

    # please input your key_api and url.
    api = args.llm_api
    url = args.llm_url
    CDLLM = LossLLM(model= llm_mode, api = api, url=url, mode = data_type)
    CDEC = ECforAlg(model=CDLLM, trainingtool=training_tool)

    print('Training Start.')

    CDEC.run_Inital()
    CDEC.run_Evolution()

    print('Training Finished.')
    
if __name__ == '__main__':

    main()









