# LifeGPT

LifeGPT is a repository containing scripts and data pertaining to a decoder-only generative pretrained transformer (GPT) model designed to learn Conway's Game of Life (Life), and replicated the next-game-state transition for *nearly all* initial conditions. (LifeGPT, in its current embodiemnt, does occasionally produce incorrect tokens in error.)


## Setting Up the Conda Environment

To set up the Conda environment for this project, you will use the `LifeGPT_env.yml` file provided in this repository. This file contains all the necessary dependencies and package versions to ensure the environment is consistent with what the project requires.

### Step-by-Step Guide to Create the Conda Environment

1. **Install Conda (if not already installed):**
   
   If you don't have Conda installed, you can download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Clone the Repository:**

   First, clone this GitHub repository to your local machine:

   ```bash
   git clone https://github.com/lamm-mit/LifeGPT.git
   cd LifeGPT

3. **Create the Conda Environment:**

    Use the LifeGPT_env.yml file to create a new Conda environment. Run the following command in your terminal:

   ```bash
    conda env create -f LifeGPT_env.yml

This command will create a new environment named LifeGPT_env with all the dependencies specified in the .yml file.

4. **Activate the Conda Environment:**

Once the environment is created, activate it using the following command:

    ```bash
    conda activate LifeGPT_env

Verify the Environment:

You can verify that the environment is set up correctly and that all necessary packages are installed by listing the packages:

bash

conda list

Remove Personal Paths:

The LifeGPT_env.yml file is configured to avoid including personal paths (e.g., prefix: lines), so no further modification is required to remove user-specific information.

Run the Project:

After activating the environment, you can run the project scripts or applications. For example:

bash

    python your_script.py

Additional Notes

    Updating the Environment:

    If there are changes in the required packages or if the environment needs to be updated, modify the LifeGPT_env.yml file accordingly, and then update the environment with:

    bash

conda env update --file LifeGPT_env.yml --prune

Deactivating the Environment:

To deactivate the environment, simply run:

bash

conda deactivate

Removing the Environment:

If you need to remove the environment at any point, use:

bash

conda remove --name LifeGPT_env --all

## Datasets

## Training
### Prewritten Scripts with Preset Training Parameters

