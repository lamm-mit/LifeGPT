# LifeGPT: Topology-Agnostic Generative Pretrained Transformer Model for Cellular Automata

The Game of Life (Life), a well known algorithm within the broader class of cellular automata (CA), exhibits complex emergent dynamics, with extreme sensitivity to initial conditions. Modeling and predicting such intricate behavior without explicit knowledge of the system's underlying topology presents a significant challenge, motivating the development of algorithms that can generalize across various grid configurations and boundary conditions. We develop a decoder-only generative pretrained transformer model to solve this problem, showing that our model can simulate Life on a toroidal grid with no prior knowledge on the size of the grid, or its periodic boundary conditions (LifeGPT). LifeGPT is topology-agnostic with respect to its training data and our results show that a GPT model is capable of capturing the deterministic rules of a Turing-complete system with near-perfect accuracy, given sufficiently diverse training data. We also introduce the idea of an `autoregressive autoregressor' to recursively implement Life using LifeGPT. Our results pave the path towards true universal computation within a large language model (LLM) framework, synthesizing of mathematical analysis with natural language processing, and probing AI systems for situational awareness about the evolution of such algorithms without ever having to compute them. Similar GPTs could potentially solve inverse problems in multicellular self-assembly by extracting CA-compatible rulesets from real-world biological systems to create new predictive models, which would have significant consequences for the fields of bioinspired materials, tissue engineering, and architected materials design.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/SbfQH-6_ZUHMgmr60BVw-.png)

# Installation and code use

# Inference with trained weights

![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/AMKWGJXj4psBwaJ5ZCzs7.png)

# Training
![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/k6JawabkK4vTWlHkCda8E.png)

# Miscalleneous 

```bibtex
@article{berkovich2024lifegpt,
  title={LifeGPT: Topology-Agnostic Generative Pretrained Transformer Model for Cellular Automata},
  author={Berkovich, Jaime A. and Buehler, Markus J.},
  journal={Preprint},
  year={2024},
}
```
