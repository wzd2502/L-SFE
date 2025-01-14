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


## Running the tests

```
python main.py --n_vars 10 --m_samples 1000 --data_type discrete --graph_type randomgraph --noise_type exp --train_method HC  --seed 627 --llm_api "your LLM API" --llm_url "Your LLM URL"
```

