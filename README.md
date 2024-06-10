# GNN_MM
A Multimodal Approach to Optimize Molecular Representations for Graph-Based Property Predictions


# Code Instructions

## Main Features
- Utilization of both global molecular features and graphical features at the node (atomic) and edge (bond) level.
- Bayesian optimization for SOAP hyperparameter tuning.
- Ability to include additional molecular, node, or edge features.
- CUDA support for accelerated computation.

## Environment Setup
- It is recommended to run the code in a Python virtual environment or a Conda environment (general instructions provided below) to manage dependencies efficiently.
- The necessary modules are listed in a requirements text file provided in the repository. Install them using `pip install -r requirements.txt`.
- Data should be organized in a `data` directory, containing:
  - A CSV file with target values named `targets.csv`.
  - A `xyz` folder with atomic coordinates for each molecule, where filenames correspond to entries in the CSV file.
    
### General CUDA setup for Linux

Create a conda environment for PyTorch using `torch-env.yml`:

```bash
conda env create -f torch-env.yml
conda activate torch-env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install botorch -c pytorch -c gpytorch -c conda-forge
conda install pyg -c pyg
```

### Data CSV Format
The CSV file should have three columns: `Name`, `Target`, `SMILES`, where each row represents a molecule with its corresponding xyz filename, target value, and SMILES string.

## Model Architecture Parameters
- `node_units`: Sets the sizes of the initial linear transformation on node features and the hidden state of the GRU. The last two values determine the dimensions of the MLP layers for final predictions.
- `edge_units`: Configures the dimensions of the edge neural network layers used in convolutional message passing.
- `molecule_units`: Specifies the dimensions of the MLP that processes molecular features before concatenation with the graph representation.
- `number_message_passes` and `processing_steps`: Control the complexity of the model by adjusting the number of message passes and the processing steps in the Set2Set layer.
- `dropout`: Sets the dropout rate within the GNN to mitigate overfitting.
- `learning_rate`, `n_epochs`, `patience`: Define the learning rate, maximum training epochs, and the patience for early stopping, respectively.
- `batch_size`: Determines the batch size for training the model.

## Molecular Feature Parameters
- `node_feature_set`, `edge_feature_set`, `molecule_feature_set`: Define the types of features (one or more of `categorical`, `continuous`, `SOAP`, `RDKit`) used for nodes, edges, and molecules.
- `desc_names`: Lists the RDKit descriptors to be used. Default features include molecular weight, atom counts, topological polar surface area (TPSA), and more.

## Bayesian Optimization Parameters
- `bounds_norm`: Normalized bounds for SOAP parameters, typically ranging from 0 to 1.
- `bounds_unnorm`: Actual value ranges for SOAP hyperparameters and the number of message passes.
- `sample_size`: The number of initial evaluations for creating the prior distribution before starting the Bayesian optimization loop.

## Running the Code
1. Set feature types for nodes, edges, and molecules as needed. For example, to use SOAP and categorical features for nodes: `node_feature_set = ["categorical", "SOAP"]`.
2. Configure `bounds_unnorm` for the SOAP hyperparameters and message passing.
3. Use the provided Slurm batch script template below, adjusting it according to your computational resources and job requirements:

```bash
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1

module purge
module load GCC/11.3.0 OpenMPI/4.1.4
module load PyTorch/1.12.1-CUDA-11.7.0
source ~/pyg/torch-env/bin/activate
srun python gnn.py
```

