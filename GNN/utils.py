"""
This module contains the helper functions for data manipulation.
"""

from chemlibs import *

def canonical_smiles(smiles):
    """
    Converts a list of SMILES strings to their canonical forms.

    Args:
        smiles (list of str): A list of SMILES strings.

    Returns:
        list of str: A list containing the canonical SMILES strings.
    """
    can_smiles=[]
    for s in smiles:
        can_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
    return can_smiles


def get_rdkit_descriptor_list():
    """
    Returns a list of the names of all descriptors from the RDKit Descriptor function. 
    Note that descriptor with index 40, Ipc, tends to yield NaN values. 
    """
    rdkit_descriptors = []
    for desc in Descriptors._descList:
        rdkit_descriptors.append(desc[0])
    return rdkit_descriptors


def get_molecular_features(SMILES, rdk_descriptors, missingVal=None):
    from rdkit.Chem import Descriptors
    import traceback
    molecule_features = []
    for smi in SMILES:
        molecule=Chem.AddHs(Chem.MolFromSmiles(smi))
        features=[]
        for desc in rdk_descriptors:
            try:
                val=getattr(Descriptors, desc)(molecule)
            except:
                traceback.print_exc()
                val=missingVal
            features.append(val)
        molecule_features.append(features)
    return np.array(molecule_features).astype(float)


class MolecularDataset(torch.utils.data.Dataset):
    '''
    Molecular feature dataset
    '''
    def __init__(self, X, y):
        if torch.is_tensor(X): 
            self.X = X
        else:
            self.X = torch.tensor(X, dtype = torch.float32)
        if torch.is_tensor(y): 
            self.y = y
        else:
            self.y = torch.tensor(y, dtype = torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    

def get_classes(targets, classes=[-4, -2, 0]):
    """
    Classifies target values into categories based on provided class boundaries.

    Args:
        targets (list of float): A list of target values to classify.
        classes (list of int, optional): The boundaries for classification. Defaults to [-4, -2, 0].

    Returns:
        list of int: A list of classification categories for each target.
    """
    classification=[]
    for target in targets:
        if target<classes[0]:
            classification.append(0)
        elif classes[0]<=target<classes[1]:
            classification.append(1)
        elif classes[1]<=target<classes[2]:
            classification.append(2)
        elif classes[2]<=target:
            classification.append(3)
    return classification


def unnormalize_parameters(x, bounds, int_indices = [1, 2, 4]):
    """
    Unnormalizes parameters from the range [0, 1] and rounds certain values to the nearest integer.

    Args:
        x (torch.Tensor): Normalized parameters.
        bounds (list of tuple): Bounds for each parameter.
        int_indices (list of indices): Indices of parameters that are integer valued and will be rounded to whole values.

    Returns:
        torch.Tensor: The unnormalized parameters.
    """
    bounds = torch.tensor(bounds)
    unnormalized_x = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    for i in int_indices:
        unnormalized_x[i] = torch.round(unnormalized_x[i])
    
    return unnormalized_x


def standardize(values, vmin, vmax, transform=1):
    """
    Standardizes the values using min-max scaling or its inverse transformation.

    Args:
        values (iterable): The values to be standardized.
        vmin (float): The minimum value for scaling.
        vmax (float): The maximum value for scaling.
        transform (int): Indicates the direction of scaling. 1 for standard scaling, -1 for inverse. Defaults to 1.
        method (str): The scaling method to use. Currently supports only "minmax".

    Returns:
        iterable: The standardized values.
    """
    c = 2.0/(vmax-vmin)
    if transform==1:
        values_new = (values - vmin)*c - 1.0
    elif transform==-1:
        values_new = (values+1.0)/c + vmin
    return values_new


def normalize_parameters(x, bounds):
    """
    Normalizes parameters to the range [0, 1].

    Args:
        x (torch.Tensor): The parameters to be normalized.
        bounds (list of tuple): Bounds for each parameter.

    Returns:
        torch.Tensor: The normalized parameters.
    """
    bounds = torch.tensor(bounds)
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    
def get_length_SOAP(soap_params,
                    xyz,
                    neighbors = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53],
                    compression = None):
    """
    Calculates the length of the SOAP (Smooth Overlap of Atomic Positions) descriptor.

    Args:
        soap_params (str): Parameters for SOAP calculation.
        xyz (iterable): XYZ coordinates of the molecule.
        neighbors (list of int): List of atomic numbers to consider as neighbors. Defaults to [1, 6, 7, 8, 9, 15, 16, 17, 35, 53].
        compression (str, optional): Compression method, if any.

    Returns:
        int: The length of the SOAP descriptor.
    """
    test_soap = compute_SOAP(xyz,
                             soap_params=soap_params,
                             neighbors=neighbors,
                             compression=compression)
    return len(test_soap[0][0])


def get_skf_ids(y, seed=333):
    """
    Generates train and test indices for stratified k-fold cross-validation.

    Args:
        y (iterable): Target values for classification.
        seed (int): Random seed for reproducibility. Defaults to 333.

    Returns:
        list of tuples: Each tuple contains train and test indices for a fold.
    """
    yclass=get_classes(y)
    X = np.random.randint(0,100,len(y))

    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    skf.get_n_splits(X,yclass)

    skf_ids=[]
    for (train_index, test_index) in skf.split(X, yclass):
        skf_ids.append((train_index, test_index))
    return skf_ids


def print_cross_validation_start(i, new_param, soap_length):
    """
    Prints the start of a cross-validation fold with SOAP parameters.

    Args:
        i (int): The index of the current fold.
        new_param (tuple): Parameters for the SOAP calculation.
        soap_length (int): The length of the SOAP descriptor.
    """
    a,b,c,d,e = new_param
    b,c,e=int(b),int(c),int(e)
    out1 = f"##### Scoring SOAP {i} of length {soap_length}: "
    out2 = "r_cut={:.3f}, ".format(a)
    out3 = f"l_max={b}, n_max={c}, "
    out4 = "sigma={:.3f}, mps={} ##### \n".format(d,e)
    print(out1+out2+out3+out4)


def gpu_to_cpu(gpu_state):
    """
    Transfer PyTorch model weights from GPU to CPU 

    Args: 
        gpu_state (dict): The weights of the model that are on GPU.

    Returns:
        cpu_state (dict): The weights of the model on CPU.
    """
    cpu_state = {key: value.to('cpu') for key, value in gpu_state.items()}
    return cpu_state


def compile_results(scores, normal_params, param_bounds, param_names):
    '''
    Get a pandas dataframe of Bayesian optimization results
    Inputs should be PyTorch tensors except param_names which should be a list
    '''
    columns = param_names + ["validation_loss", "train_loss"]
    unnormal_params = np.array([unnormalize_parameters(param, param_bounds).numpy() for param in normal_params])
    results_df = pd.DataFrame(data=np.hstack([unnormal_params, scores.numpy()]),
                              columns=columns)
    for param in param_names:
        if param in ["l_max", "n_max", "message_passes"]:
            results_df[param] = results_df[param].values.astype(int)
    return results_df


def predict(model, data_loader, pyg = True):
    """
    Make predictions using the model.

    Parameters:
    - model: The trained model.
    - data_loader: DataLoader for the dataset.
    - pyg: True if DataLoader is of PyTorch Geometric or False if from torch.utils

    Returns:
    - predictions: Predicted values.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        if pyg:
            for data in data_loader:
                y_hat = model(data)
                predictions.extend(y_hat.view(-1).numpy())
        elif not pyg:
            for X, y in data_loader:
                y_hat = model(X)
                predictions.extend(y_hat.view(-1).numpy())
    return np.array(predictions)


def compute_metrics(predicted, actual, list_out = True):
    """
    Compute RMSE, MAE, Pearson correlation coefficient, R2, percentage within 50% of actual

    Parameters:
    - predicted: Predicted values.
    - actual: Actual values.

    Returns:
    - List or dictionary containing RMSE, MAE, Pearson correlation, R2 score.
    """
    rmse = mean_squared_error(actual, predicted, squared=False)
    mae = mean_absolute_error(actual, predicted)
    pearson_corr, _ = pearsonr(actual, predicted)
    r2 = r2_score(actual, predicted)
    pm50 = np.where(np.abs(predicted-actual)<0.50)[0].shape[0]/actual.shape[0]
    if list_out:
        return [rmse, mae, pearson_corr, r2, pm50]
    elif not list_out:
        return {'RMSE': rmse, 'MAE': mae, 'Pearson': pearson_corr, "r2_score": r2, "within_50": pm50}

    
def print_parameter_summary(parameters):
    print("{:<20} {}".format("parameter", "value"))
    print("{:<20} {:.3f}".format("r_cut", parameters[0]))
    print("{:<20} {:.0f}".format("l_max", parameters[1]))
    print("{:<20} {:.0f}".format("n_max", parameters[2]))
    print("{:<20} {:.3f}".format("sigma", parameters[3]))
    print("{:<20} {:.0f}".format("message passes", parameters[4]))

    
def print_score_summary(scores):
    print("{:<40} {:.5f}".format("Root Mean Squared Error:", scores[0]))
    print("{:<40} {:.5f}".format("Mean Absolute Error:", scores[1]))
    print("{:<40} {:.5f}".format("Pearson Correlation Coefficient (r):", scores[2]))
    print("{:<40} {:.5f}".format("Coefficient of Determination (R2):", scores[3]))
    print("{:<40} {:.5f}".format("% within 0.50 of experimental:", scores[4]))


def plot_scores(scores, 
                score_names = ["RMSE", "MAE", "PCC", "$R^2$", "%0.50"],
                x0=-2.5, x1=1.75,
                y=[-9.25, -10, -10.75, -11.5, -12.25],
                fontsize=12):
    for i in range(len(scores)):
        plt.text(x0+x1,y[i]-1,f"{score_names[i]}", fontsize=fontsize)
        plt.text(x1,y[i]-1,f"{scores[i]:.3f}", fontsize=fontsize)
        
        
# def plot_scores(scores, 
#                 x=-5.5,
#                 y=[-9.25,-10,-10.75,-11.5,-12.25],
#                 fontsize=12):
#     plt.text(x,y[0],f"RMSE:  {scores[0]:.3f}", fontsize=fontsize)
#     plt.text(x,y[1],f"MAE:   {scores[1]:.3f}", fontsize=fontsize)
#     plt.text(x,y[2],f"PCC:   {scores[2]:.3f}", fontsize=fontsize)
#     plt.text(x,y[3],f"R2:    {scores[3]:.3f}", fontsize=fontsize)
#     plt.text(x,y[4],f"%0.50: {scores[4]:.3f}", fontsize=fontsize)


def format_scatter_plot(lb, ub,
                        linewidth=1.5, diagwidth=1.0,
                        axis_labels = None,
                        legend_size = None,
                        fontsize = 14):
    plt.legend(prop={'size':10})
    plt.plot(np.linspace(lb,ub,100),np.linspace(lb,ub,100),
             color='darkgrey',zorder=0,lw=linewidth)
    plt.plot(np.linspace(lb,ub,100),np.linspace(lb,ub,100)+diagwidth,
             color='lightgrey',zorder=0,lw=linewidth)
    plt.plot(np.linspace(lb,ub,100),np.linspace(lb,ub,100)-diagwidth,
             color='lightgrey',zorder=0,lw=linewidth)
    plt.xlim(lb,ub)
    plt.ylim(lb,ub)
    if legend_size is not None:
        plt.legend(prop={'size': legend_size})
    if axis_labels is not None:
        plt.xlabel(axis_labels[0], fontsize=fontsize)
        plt.ylabel(axis_labels[1], fontsize=fontsize)
    else:
        plt.xlabel('Experimental',fontsize=fontsize)
        plt.ylabel('Predicted',fontsize=fontsize)



def turnoff_upperaxis(fig, ncols=4):
    for i in range(ncols):
        for j in range(i+1,ncols):
            fig.axes[i][j].axis('off')


def plot_labels(fig, labels, fs=20):
    for i in range(len(labels)):
        fig.axes[i][0].set_ylabel(labels[i], fontsize = fs)
        fig.axes[len(labels)-1][i].set_xlabel(labels[i], fontsize = fs)


def get_cbar_ticks(loss_values):
    x1 = Decimal('{}'.format(loss_values.min()))
    x3 = Decimal('{}'.format(loss_values.max()))
    x1 = float(x1.quantize(Decimal('0.001'), ROUND_UP))
    x3 = float(x3.quantize(Decimal('0.001'), ROUND_DOWN))
    x2 = round((x1+x3)/2,3)
    return [x1,x2,x3]


def plot_bayesopt_results(unnormal_params,cmap,loss_values,
                          labels,fs=18,title=None,loss="Validation loss",
                          savefile=None,alpha=0.8,bins=9,levels=100):
    grid = sns.PairGrid(unnormal_params, corner = False)
    turnoff_upperaxis(grid, len(labels))

    grid.map_diag(sns.histplot, fill=True, linewidth=1, bins=bins)
    grid.map_lower(sns.kdeplot, levels=levels, color="blue", alpha = 0.7, fill = True, cmap = 'Blues')
    grid.map_lower(sns.scatterplot,
                    marker = 'o',
                    hue=loss_values,
                    s=60,
                    palette=cmap,
                    linewidth = 1,
                    alpha = alpha,)

    ticks = get_cbar_ticks(loss_values)
    norm = plt.Normalize(loss_values.min(), loss_values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(ax=grid.axes[0,len(labels)-1],
                        orientation='horizontal',
#                         location = "top",
                        mappable=sm,
                        ticks=ticks,)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=fs*0.9)
    cbar.set_label(loss, size = fs*0.95, labelpad = 10)
    plot_labels(grid, labels, fs)
    plt.tight_layout()
    if title is not None:
        grid.fig.subplots_adjust(top=0.925)
        grid.fig.suptitle(title, fontsize = fs*1.05)
    if savefile is not None:
        plt.savefig(savefile, dpi=300)
    plt.show()
