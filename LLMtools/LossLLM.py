from openai import OpenAI
from ECtools.heuristic import Alg

class LossLLM:
    def __init__(self, model, api, url, mode):

        if len(api) == 0:
            raise ValueError("Initialization failed: api and url cannot be empty")
        self.model = model
        self.client = OpenAI(
                api_key=api,
                base_url =url
            )

        
        if mode == 'continuous':
            self.mode = 'continuous'
            self.std = """
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
        elif mode == 'discrete':
            self.mode = 'discrete'
            self.std = """
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
    
# def subscoreLLM(child, parents, Data):
#     '''
#     Idea: This function uses of the BIC score to measure the goodness-of-fit of a subgraph to a discrete / continuous dataset.

    
#     Input and Output:
#     child: an index;
#     parents: a list of index, it is [] when child is a root node;
#     Data: a m*n numpy, where m is the number of instances, and n is the numer of variables.
#     score: the fitness of subgrapg parents -> child.
#     '''

#     # The code implementation.

#     return score
    def Initilization(self):
        '''
        :return: code implementation. str
        '''

        prompt = (f"""
Please help design a new score function to measure how well the subgraph Pa_i -> i fits the {self.mode} observation dataset. Below is as example:
\n {self.std}. 
Follow these steps to complete the task:
1. Understand the BIC Score: The demo uses the Bayesian Information Criterion (BIC) score for the subgraph. Make sure you understand how it works.
2. Modify to Prevent Overfitting: Think about how to adjust the algorithm to reduce overfitting and improve the graphical accuracy of the results. You can experiment with different parameters or try a new fitting approach. Write a brief summary of your idea in one sentence and add it as a comment at the start of the code.
3. Implement Your Solution: Write the code to apply your changes. Ensure the function name, input, and output are the same as in the demo code. Only provide the code—no explanations needed.
""")
        code = self.get_completion(prompt)
        return code

    def CrossOver(self, Alg1:Alg, Alg2: Alg):
        '''
        find a new Alg based on given alg1 and alg2.
        :param Alg1: the Alg object
        :param Alg2: the Alg object
        :return: the code, type of new Alg object
        '''
        prompt = (f"""
Please help design a new score function to measure how well the subgraph Pa_i -> i fits the {self.mode} observation dataset. Here are two examples and their performance:
Algorithm 1: {Alg1.code};
loss of Algorithm 1: {Alg1.loss};
Algorithm 2: {Alg2.code};
loss of Algorithm 1: {Alg2.loss}. 
Note: A lower loss means better performance of the algorithm. Follow these steps to complete the task:
1. Read the code comments: Understand how each of the two algorithms calculates the score for the subgraph.
2. Review the code: Analyze the reasons behind the good or poor performance of the two algorithms.
3. Design a new algorithm: Think of a new algorithm that combines ideas from both, improving their effectiveness. Summarize your idea in one sentence and add it to the beginning of the code comments.
4. Implement your solution: Write the code to implement your new idea. Ensure the function name, input, and output match the demo code. Provide only the code—no explanations are needed.
""")

        code = self.get_completion(prompt)
        return code

    def NewAdd(self, Algs:list):
        '''
        create new algs that differ from previous.

        '''
        output_strings = []
        restriction = 2
        for i in range(restriction):
            my_str = f"The code of Algorithm {i}: '{Algs[i].code}';"
            output_strings.append(my_str)

        result = "\n".join(output_strings)
        prompt = (f"""
Please help design a new score function to measure how well the subgraph Pa_i -> i fits the {self.mode} observation dataset. Here are some examples: {result}.
Note: Follow these steps to complete the task:
1. Read the code comments: Understand how the current algorithm calculates the score for the subgraph.
2. Design a new algorithm: Think of a new approach that is different from the demo in both structure and concept. Write a one-sentence summary of your idea and add it to the beginning of the code comments.
3. Implement your solution: Write the code to implement your new idea. Make sure the function name, input, and output match the demo code. Provide only the code—no explanations are needed.
""")

        code = self.get_completion(prompt)
        return code

    def Mutation1(self, Alg1: Alg):
        '''
        find a new Alg based on given Alg1.
        :param Alg1: the Alg object
        :return: the code, type of new Alg object
        '''
        prompt = (f"""
Please help design a new score function to measure how well the subgraph Pa_i -> i fits the {self.mode} observation dataset. Here is an examples and its performance: 
Algorithm 1: {Alg1.code};
loss of Algorithm 1: {Alg1.loss}.
Note: A lower loss indicates better performance of the algorithm. Follow these steps to complete the task:
1. Read the code comments: Understand how the algorithm calculates the score for the subgraph.
2. Review the code: Analyze why the algorithm performs well or poorly.
3. Improve the algorithm: Think of ways to modify the algorithm for better performance. Summarize your idea in one sentence and add it to the beginning of the code comments.
4. Implement your solution: Modify the code according to your idea. Ensure that the function name, input, and output remain the same as in the demo code. Provide only the code—no explanations are needed.
""")
        code= self.get_completion(prompt)
        return code
    
    def Mutation2(self, Alg1: Alg):
        '''
        find a new Alg based on given Alg1.
        :param Alg1: the Alg object
        :return: the code, type of new Alg object
        '''

        prompt = (f"""
Please help design a new score function to measure how well the subgraph Pa_i -> i fits the {self.mode} observation dataset. Here is an examples and its performance: 
Algorithm 1: {Alg1.code};
loss of Algorithm 1: {Alg1.loss}.
Note: The lower the loss, the better the algorithm. Follow these steps to complete the task:
1. Read the code comments: Understand how the algorithm calculates the score for the subgraph.
2. Review the code: Identify why the algorithm performs well or poorly.
3. Optimize performance: Think about how to adjust parameters for better performance. Summarize your approach in one sentence and add it to the code comments.
4. Implement your solution: Modify the code to reflect your improvements. Ensure the function name, input, and output match the original demo code. Provide only the code—no explanations needed.
""")
        code= self.get_completion(prompt)
        return code

    def get_completion(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content


