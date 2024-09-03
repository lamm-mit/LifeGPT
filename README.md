=======
# LifeGPT: Topology-Agnostic Generative Pretrained Transformer Model for Cellular Automata

The Game of Life (Life), a well known algorithm within the broader class of cellular automata (CA), exhibits complex emergent dynamics, with extreme sensitivity to initial conditions. Modeling and predicting such intricate behavior without explicit knowledge of the system's underlying topology presents a significant challenge, motivating the development of algorithms that can generalize across various grid configurations and boundary conditions. We develop a decoder-only generative pretrained transformer model to solve this problem, showing that our model can simulate Life on a toroidal grid with no prior knowledge on the size of the grid, or its periodic boundary conditions (LifeGPT). LifeGPT is topology-agnostic with respect to its training data and our results show that a GPT model is capable of capturing the deterministic rules of a Turing-complete system with near-perfect accuracy, given sufficiently diverse training data. We also introduce the idea of an `autoregressive autoregressor' to recursively implement Life using LifeGPT. Our results pave the path towards true universal computation within a large language model (LLM) framework, synthesizing of mathematical analysis with natural language processing, and probing AI systems for situational awareness about the evolution of such algorithms without ever having to compute them. Similar GPTs could potentially solve inverse problems in multicellular self-assembly by extracting CA-compatible rulesets from real-world biological systems to create new predictive models, which would have significant consequences for the fields of bioinspired materials, tissue engineering, and architected materials design.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/SbfQH-6_ZUHMgmr60BVw-.png)

# Installation and code use


## Setting Up the Conda Environment

To set up the Conda environment for this project, you will use the `LifeGPT_env.yml` file provided in this repository. This file contains all the necessary dependencies and package versions to ensure the environment is consistent with what the project requires.

### Step-by-Step Guide to Create the LifeGPT Conda Environment

1. **Install Conda (if not already installed):**
   
   If you don't have Conda installed, you can download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Clone the Repository:**

   First, clone this GitHub repository to your local machine:

   ```bash
   git clone https://github.com/lamm-mit/LifeGPT.git
   cd LifeGPT
   ```

3. **Create the Conda Environment:**

    Use the LifeGPT_env.yml file to create a new Conda environment. Run the following command in your terminal:

   ```bash
    conda env create -f LifeGPT_env.yml
    ```

This command will create a new environment named LifeGPT_env with all the dependencies specified in the .yml file.

4. **Activate the Conda Environment:**

    Once the environment is created, activate it using the following command:

    ```bash
    conda activate LifeGPT_env
    ```
5. **Verify the Environment:**

    You can verify that the environment is set up correctly and that all necessary packages are installed by listing the packages:

    ```bash
    conda list
    ```

6. **Updating the Environment:**

    If there are changes in the required packages or if the environment needs to be updated, modify the LifeGPT_env.yml file accordingly, and then update the environment with:

    ```bash
    conda env update --file LifeGPT_env.yml --prune
    ```

7. **Deactivating the Environment:**

    To deactivate the environment, simply run:

    ```bash
    conda deactivate
    ```

8. **Removing the Environment:**

    If you need to remove the environment at any point, use:
    
    ```bash
    conda remove --name LifeGPT_env --all
    ```

## Datasets
Included in this repository, in the subfolder [LifeGPT](https://github.com/lamm-mit/LifeGPT/tree/main/LifeGPT), are several csv files corresponding to training, validation, and testing data.
1. **Testing Data:**
    Both testing data files ([conway_test_states_32by32_20240716_151502.csv](https://github.com/lamm-mit/LifeGPT/blob/main/LifeGPT/conway_test_states_32by32_20240716_151502.csv) and [conway_test_states_32by32_20240827_172807.csv](https://github.com/lamm-mit/LifeGPT/blob/main/LifeGPT/conway_test_states_32by32_20240827_172807.csv)) correspond to the same initial conditions (ICs). They are distinct because [conway_test_states_32by32_20240716_151502.csv](https://github.com/lamm-mit/LifeGPT/blob/main/LifeGPT/conway_test_states_32by32_20240716_151502.csv) contains a total of 10 timesteps for Life, while [conway_test_states_32by32_20240827_172807.csv](https://github.com/lamm-mit/LifeGPT/blob/main/LifeGPT/conway_test_states_32by32_20240827_172807.csv) contains 250 timesteps. These data are used for accuracy benchmarking (see [Accuracy_Benchmarking.ipynb](https://github.com/lamm-mit/LifeGPT/blob/main/Accuracy_Benchmarking.ipynb)), as well as for comparison with 'ground truth' (GT) in the case of the 'autoregressive autoregressor' (ARAR) -- see [ARAR_249_iterations.ipynb](https://github.com/lamm-mit/LifeGPT/blob/main/ARAR_249_iterations.ipynb) and [ARAR_9_iterations.ipynb](https://github.com/lamm-mit/LifeGPT/blob/main/ARAR_9_iterations.ipynb).

2. **Training and Validation Data**
    In addition, there are 2 files corresponding to training data, and 2 files corresponding to validation data. The data can be divided into two groups high-entropy and broad-entropy.
    
    a. *High-Entropy*
        High-entropy data is defined as data which contains stochastically generated ICs, where there is an equal expected value of finding live (1-state) or dead (0-state) cells. We include a 10000-sample high-entropy dataset for training ([conway_states_0.5_0.5_10000by32by32by10_toroidal_20240813_224815_sorder0.5_eorder0.5.csv](https://github.com/lamm-mit/LifeGPT/blob/main/LifeGPT/conway_states_0.5_0.5_10000by32by32by10_toroidal_20240813_224815_sorder0.5_eorder0.5.csv)) and a 1000-sample high-entropy dataset for finding validation losses ([conway_states_0.5_0.5_1000by32by32by10_toroidal_20240813_224611_sorder0.5_eorder0.5.csv](https://github.com/lamm-mit/LifeGPT/blob/main/LifeGPT/conway_states_0.5_0.5_1000by32by32by10_toroidal_20240813_224611_sorder0.5_eorder0.5.csv)). 

    b. *Broad-Entropy*
        Broad-entropy data is defined as data which contains stochastically generated ICs, where each sample in the dataset is generated with an order parameter ranging linearly from 0 to 1, where the order parameter may be interpreted as the expected value of the probability of finding a 1 in the corresponding IC. We include a 10000-sample broad-entropy dataset for training ([conway_states_0_1_10000by32by32by10_toroidal_20240711_133408.csv](https://github.com/lamm-mit/LifeGPT/blob/main/LifeGPT/conway_states_0_1_10000by32by32by10_toroidal_20240711_133408.csv)) and a 1000-sample broad-entropy dataset for finding validation losses ([conway_states_0_1_1000by32by32by10_toroidal_20240711_151806.csv](https://github.com/lamm-mit/LifeGPT/blob/main/LifeGPT/conway_states_0_1_1000by32by32by10_toroidal_20240711_151806.csv)).

3. **Example Data**
    For demonstrative purposes, we also include two datasets ([EXAMPLE_conway_states_0_1_100by50by50by20_toroidal_20240821_152545.csv](https://github.com/lamm-mit/LifeGPT/blob/main/LifeGPT/EXAMPLE_conway_states_0_1_100by50by50by20_toroidal_20240821_152545.csv) and [EXAMPLE_conway_states_0_1_100by50by50by20_toroidal_20240821_152920.csv](https://github.com/lamm-mit/LifeGPT/blob/main/LifeGPT/EXAMPLE_conway_states_0_1_100by50by50by20_toroidal_20240821_152920.csv))




## Training
### Prewritten Scripts with Preset Training Parameters
Included in this repository are two .py files containing prewritten code for training LifeGPT models using different training data and hyperparameters.

The file [LifeGPT_toroidal_rot_pos_on_no_forgetful_mask_high_ent.py]()

# Inference with trained weights
Accuracy benchmarking is done in the Jupyter Notebook [Accuracy_Benchmarking.ipynb](https://github.com/lamm-mit/LifeGPT/blob/main/Accuracy_Benchmarking.ipynb). 


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
>>>>>>> 2dd743f2da611e0a40f0797fec8cfddde1ca7b9a
