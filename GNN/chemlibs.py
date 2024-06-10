# general 
import numpy as np
import pickle as pkl
import pandas as pd
import collections
from itertools import combinations
import copy
from copy import deepcopy
import os
import json
import gc
from time import time
from decimal import Decimal, ROUND_UP, ROUND_HALF_UP, ROUND_DOWN
import random


# visualization
import matplotlib.pyplot as plt
import seaborn as sns


# machine learning and statistics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler, MaxAbsScaler
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# deep learning
import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential, BatchNorm1d, Module 
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import MSELoss
torch.set_printoptions(sci_mode=False)

# Bayesian optimization
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin
from botorch.utils.sampling import draw_sobol_samples


# chemistry
from ase import Atoms
from ase.io import read, write
import ase.build
#from quippy import descriptors
from rdkit import Chem
#from rdkit import RDLogger
#from rdkit.Chem.Draw import IPythonConsole
#from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import AllChem,Crippen,Descriptors
from dscribe import descriptors

from utils import *

# Temporary suppress warnings and RDKit logs
#import warnings
# warnings.filterwarnings("ignore")
# RDLogger.DisableLog("rdApp.*")

# dont want scientific notation
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float':"{0:0.5f}".format})


def get_fc_layers(layer_units,
                 activation = ReLU()):
    layers = []
    for i in range(1, len(layer_units)):
        layers.append(Linear(layer_units[i-1], layer_units[i]))
        layers.append(BatchNorm1d(layer_units[i]))
        layers.append(activation)
    return layers    


class GraphNet(torch.nn.Module):
    """
    A PyTorch implementation of a Graph Neural Network (GNN) for molecular graph analysis, specifically of the Graph Convolution type.

    The network uses linear layers, batch normalization, message passing with GCN, GRU, and Set2Set layers to process graph data for predictions.

    Attributes:
        node_units (list): Number of units in each node layer.
        edge_units (list): Number of units in each edge layer.
        molecule_units (list): Number of units in each molecule layer.
        number_message_passes (int): Number of message passing iterations.
        dropout (list): Dropout rates.
        processing_steps (int): Number of processing steps for Set2Set layer.
    """
    def __init__(self,
                 node_units,
                 edge_units,
                 molecule_units = [0],
                 fcnn_units = [256, 128, 32],
                 number_message_passes = 4,
                 dropout = [0.50, 0.50],
                 processing_steps = 5,
                 ):
        super().__init__()

        self.node_units = node_units
        self.edge_units = edge_units
        self.molecule_units = molecule_units
        self.fcnn_units = fcnn_units
        self.number_message_passes = number_message_passes
        self.processing_steps = processing_steps

        # Linear input layers for atom_features
        self.node_lin0 = Linear(node_units[0], node_units[1])
        self.node_bn0 = BatchNorm1d(node_units[1])

        # FCNN input for molecule features (optional)
        if (molecule_units[0]>0) & (len(molecule_units)>1):
            self.mol_lin0 = Linear(molecule_units[0], molecule_units[1])
            self.mol_bn0 = BatchNorm1d(molecule_units[1])

        # Message passing using GCN
        edge_layers = get_fc_layers(edge_units)
        edge_layers.append(Linear(edge_units[-1], node_units[-1]*node_units[-1]))
        edge_nn = Sequential(*edge_layers)
        self.conv = NNConv(node_units[-1], node_units[-1], edge_nn, aggr='mean')
        self.gru = GRU(node_units[-1], node_units[-1])
        self.dropout1 = torch.nn.Dropout(dropout[0])        

        # Readout to obtain graph-level embedding, Output has length 2*input_dime
        self.set2set = Set2Set(node_units[-1], processing_steps=processing_steps)

        fcnn_units = [2*node_units[-1]+molecule_units[-1]]+fcnn_units
        fcnn_layers = get_fc_layers(fcnn_units)
        fcnn_layers.append(torch.nn.Linear(fcnn_units[-1], 1))
        self.fcnn = torch.nn.Sequential(*fcnn_layers) 
     

    def forward(self, data):
        """
        Defines the forward pass of the GraphNet model.

        Args:
            data: Data object containing the graph data (nodes, edges, etc.).

        Returns:x
            Tensor: The output tensor of the network.
        """
        # Applying graph layers to node/edge features to obtain graph-level representation
        xout = self.node_bn0(self.node_lin0(data.x))
        xout = F.relu(xout)
        h = xout.unsqueeze(0)
        
        for i in range(self.number_message_passes):
            m = F.relu(self.conv(xout, data.edge_index, data.edge_attr))
            xout, h = self.gru(m.unsqueeze(0), h)
            xout = xout.squeeze(0)
            xout = self.dropout1(xout)
        xout = self.set2set(xout, data.batch)

        # Process molecule features (if present) through 2-layer MLP and concatenate with graph representation
        if (self.molecule_units[0]>0) & (len(self.molecule_units)>1):
            mol_out = self.mol_bn0(self.mol_lin0(data.molecule_features))
            mol_out = F.relu(mol_out)
            xout = torch.cat((xout, mol_out), 1)
        elif self.molecule_units[0] > 0:
            xout = torch.cat((xout, data.molecule_features), 1)

        # FCNN for solubility prediction
        xout = self.fcnn(xout)

        return xout.view(-1)
    

def train_model(model,
                train_loader,
                validation_loader,
                epochs=300,
                patience=50,
                learning_rate=0.001,
                device="cpu",
                verbose=1):
    """
    Trains the GraphNet model with early stopping.

    Args:
        train_loader: DataLoader for the training data.
        validation_loader: DataLoader for the validation data.
        epochs (int): Number of epochs to train, defaults to 300.
        patience (int): Patience for early stopping, defaults to 50.
        verbose (int): Verbosity level, defaults to 1.

    Returns:
        tuple: Best state dictionary, best validation loss, best training loss.
    """

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = MSELoss()
    
    best_train_loss, best_val_loss = float('inf'), float('inf')
    best_state_dict = None
    counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        # train_loss = evaluate_epoch(train_loader, optimize_on = True)
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            y_hat = model(batch)
            
            loss = loss_function(y_hat, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Sum up batch loss
            train_loss += loss.item() 
        # Average loss across batches
        train_loss = train_loss / len(train_loader)        

        # Validation
        model.eval()
        with torch.no_grad():  # No gradients needed for validation
            # val_loss = evaluate_epoch(validation_loader, optimize_on = False)
            val_loss = 0.0
            for batch in validation_loader:
                batch = batch.to(device)
                y_hat = model(batch)
                
                loss = loss_function(y_hat, batch.y)
                # Sum up batch loss
                val_loss += loss.item() 
            # Average loss across batches
            val_loss = val_loss / len(validation_loader)       

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss, best_train_loss = val_loss, train_loss
            best_state_dict = model.state_dict()
            counter = 0
            if verbose:
                print(f"Epoch {epoch+1}: Improved validation loss {val_loss:.3f} achieved.")
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"Early stopping after {patience} epochs without improvement.")
                break

        # if verbose and (epoch + 1) % 10 == 0:  # Print every 10 epochs.
        if verbose:
            print(f"Epoch {epoch+1}: Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
    
        torch.cuda.empty_cache()
        gc.collect()

    return gpu_to_cpu(best_state_dict), best_val_loss, best_train_loss 


def BayesOpt_param(normal_params, prior_scores, bounds):
    """
    Performs Bayesian optimization for hyperparameter tuning.

    Args:
        normal_params: Normalized parameters for Bayesian optimization.
        prior_scores: Scores from prior evaluations.
        bounds: Bounds for the parameters.

    Returns:
        Tensor: The next set of parameters to evaluate.
    """
    # GP model and likelihood
    gp = SingleTaskGP(normal_params.double(), prior_scores.unsqueeze(1).double())
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # Acquisition function
    EI = ExpectedImprovement(gp, prior_scores.min(), maximize=False)

    # Optimize acquisition function, get new params
    new_param_unnorm, _ = optimize_acqf(acq_function=EI,
                                        bounds=bounds,
                                        q=1,
                                        num_restarts=10,
                                        raw_samples=100)
    return new_param_unnorm.squeeze(0), gp


class RegNet(torch.nn.Module):
    
    def __init__(self, activation, 
                 input_units, 
                 hidden_units = [256, 256, 64]):
        """
        PyTorch implemention of a fully-connected neural network.
        
        Attributes:
            activation: Activation function (object) to be used as non-linearity in the network.
            input_units: Number of elements of the input. 
            hidden_units: A list of integers specifying the hidden layer units in the neural network.
            dropout: A list of proportions specficying the amount of dropout in each layer.
        """
        super().__init__()
         
        layers = get_fc_layers([input_units]+hidden_units)
        layers.append(torch.nn.Linear(hidden_units[-1], 1))
        self.fcnn = torch.nn.Sequential(*layers) 
        self.config = {"activation": activation, 
                       "input_units": input_units,
                       "hidden_units": hidden_units} 
        
    def forward(self, x):
        """
          Forward pass
        """
        return self.fcnn(x)       


def train_regnet(model, 
                 train_loader, validation_loader,
                 device = "cpu", 
                 n_epochs = 300,
                 patience = 100,
                 learning_rate = 1e-4,
                 verbose = True):
    
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr = learning_rate)
    loss_function = MSELoss()
    best_validation_loss = np.inf

    history = []
    counter = 0
    for epoch in range(n_epochs):

        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:

            # Transfer to GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # forward pass
            y_hat = model(X_batch)
            loss = loss_function(y_hat, y_batch)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # update weights
            optimizer.step()
            train_loss += loss.item()


        # Evaluate accuracy using validation set
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():  
            for X_batch, y_batch in validation_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_hat = model(X_batch)
                loss = loss_function(y_hat, y_batch)
                validation_loss += loss.item()
        
        train_loss = train_loss/len(train_loader)
        validation_loss = validation_loss/len(validation_loader)
        history.append([train_loss, validation_loss])
        
        if verbose:
            print(f"Epoch {epoch+1}: Training loss: {train_loss:.5f}, Validation loss: {validation_loss:.5f}")
        
        # Check for improvement
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state_dict = model.state_dict()
            counter = 0
            if verbose:
                print(f"Epoch {epoch+1}: Improved validation loss {best_validation_loss:.5f} achieved.")
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"Early stopping after {patience} epochs without improvement.")
                break
    
        torch.cuda.empty_cache()
        gc.collect()

    return best_state_dict, np.array(history)


