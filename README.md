# L-SFE

LLM-enhanced Score Function Evolution for Causal Structure Learning

## Getting Started

Requirements for the software and other tools to build, test and push 
- alive_progress==3.2.0
- causal_learn==0.1.3.8
- JPype1==1.5.1
- networkx==3.4
- numpy==2.2.1
- openai==1.59.7
- pandas==2.2.3
- pgmpy==0.1.26
- tqdm==4.66.5

This project use the [pytetrad](https://www.cmu.edu/dietrich/philosophy/tetrad/) for data generation, and HC in [pgmpy](https://pgmpy.org/), BOSS and GES in [causal-learn](https://causal-learn.readthedocs.io/en/latest/index.html) for training the LLM genreated score functions. We really thanks to their contribution.


## Running the tests

```
python main.py --n_vars 10 --m_samples 1000 --data_type discrete --graph_type randomgraph --noise_type exp --train_method HC  --seed 627 --llm_type "your LLM type" --llm_api "your LLM API" --llm_url "Your LLM URL"
```
## Supplymentary Materials.
supplementary material is available on here. 

You can quickly train your own score functions use the above script.  And only the "llm_type", "llm_api" and "llm_url" are necessary. 
