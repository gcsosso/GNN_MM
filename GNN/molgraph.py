"""
This module contains the classes and functions used to create molecular graphs.
"""

from chemlibs import *


class Categorical_Featurizer:
    """
    This class is used to create a one-hot encoding of categorical features.
    """
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)
    def encode(self, item):
        """
        This function encodes a single feature into a one-hot vector.
        """
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(item)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class Continuous_Featurizer:
    """
    This class is used to create a continuous feature vector.
    """
    def __init__(self, continuous_features):
        self.continuous_features = continuous_features
        
    def encode(self, item, molecule):
        """
        This function encodes a single feature into a continuous vector.
        """
        output = []
        for name_feature in self.continuous_features:
            output += getattr(self, name_feature)(item, molecule)
        return output
    
class Atom_Categorical_Featurizer(Categorical_Featurizer):
    """
    This class is used to extract categorical features from atoms in a molecule.

    Inherits from Categorical_Featurizer class.

    Methods:
        symbol(atom): Returns the symbol of the atom.
        n_valence(atom): Returns the total valence of the atom.
        n_hydrogens(atom): Returns the number of hydrogen atoms attached to the atom.
        hybridization(atom): Returns the hybridization of the atom.
        isin_ring(atom): Checks if the atom is part of a ring structure.
        nodal_degree(atom): Returns the degree of the atom in the molecule.
        chiral_tag(atom): Returns the chiral tag of the atom.
        charge(atom): Returns the formal charge of the atom.
        aromatic(atom): Checks if the atom is aromatic.
        radical_electrons(atom): Returns the number of radical electrons of the atom.
        number_bonds(atom): Returns the number of bonds the atom has.
        atom_types(atom): Returns the type of the atom based on atomic number.
    """
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()

    def isin_ring(self, atom):
        return atom.IsInRing()

    def nodal_degree(self, atom):
        return atom.GetDegree()

    def chiral_tag(self, atom):
        chi_tag=atom.GetChiralTag()
        return chi_tag.name

    def charge(self, atom):
        return atom.GetFormalCharge()

    def aromatic(self, atom):
        return atom.GetIsAromatic()

    def radical_electrons(self, atom):
        return atom.GetNumRadicalElectrons()

    def number_bonds(self, atom):
        return len(atom.GetBonds())

    def atom_types(self, atom):
        atom_typer=dict(zip(periodic_data["AtomicNumber"].values,
                            periodic_data["Type"].values))
        return atom_typer[atom.GetAtomicNum()]


class Bond_Categorical_Featurizer(Categorical_Featurizer):
    """
    This class is used for categorical feature extraction from bonds in a molecule.

    Inherits from Categorical_Featurizer class.

    Methods:
        bond_type(bond): Returns the type of the bond.
        conjugated(bond): Checks if the bond is conjugated.
        isin_ring(bond): Checks if the bond is part of a ring structure.
        bond_has_H(bond): Checks if the bond involves a hydrogen atom.
    """
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()

    def isin_ring(self, bond):
        return bond.IsInRing()

    def bond_has_H(self, bond):
        bond_symbols=(bond.GetBeginAtom().GetSymbol(),bond.GetEndAtom().GetSymbol())
        if "H" in bond_symbols:
            return True
        elif "H" not in bond_symbols:
            return False


class Atom_Continuous_Featurizer(Continuous_Featurizer):
    """
    This class is used for extracting continuous features from atoms in a molecule.

    Inherits from Continuous_Featurizer.

    Methods:
        element_data(atom, molecule): Extracts predefined continuous features for an atom.
        mean_interatomic_distance(atom, molecule): Calculates the mean interatomic distance for an atom.
        atom_soap(atom, molecule): Retrieves SOAP (Smooth Overlap of Atomic Positions) features for an atom.
    """
    def __init__(self, continuous_features):
        super().__init__(continuous_features)

    def element_data(self, atom, molecule):
        """
        Extracts predefined continuous features for an atom. This can be edited to add/remove features.

        The features in order are: 'AtomicMass', 'MeltingPoint', 'BoilingPoint', 'AtomicRadius', 'NumberofNeutrons', and 'SpecificHeat'.

        Args:
            atom: An atom object.
            molecule: The molecule object to which the atom belongs.

        Returns:
            list: A list containing the continuous features of the atom.
        """
        atom_values={'Ge': [72.64, 1211.45, 3106.0, 1.5, 41.0, 0.32],
                    'O': [15.999, 50.5, 90.2, 0.65, 8.0, 0.918],
                    'Cl': [35.453, 172.31, 239.11, 0.97, 18.0, 0.479],
                    'I': [126.904, 386.65, 457.4, 1.3, 74.0, 0.214],
                    'N': [14.007, 63.29, 77.36, 0.75, 7.0, 1.04],
                    'Ag': [107.868, 1234.15, 2435.0, 1.8, 61.0, 0.235],
                    'K': [39.098, 336.5, 1032.0, 2.8, 20.0, 0.757],
                    'Ba': [137.327, 1002.15, 2170.0, 2.8, 81.0, 0.204],
                    'S': [32.065, 388.51, 717.8, 1.1, 16.0, 0.71],
                    'As': [74.922, 1090.15, 887.0, 1.3, 42.0, 0.329],
                    'Sn': [118.71, 505.21, 2875.0, 1.7, 69.0, 0.228],
                    'C': [12.011, 3948.15, 4300.0, 0.91, 6.0, 0.709],
                    'P': [30.974, 317.25, 553.0, 1.2, 16.0, 0.769],
                    'Hg': [200.59, 234.43, 630.0, 1.8, 121.0, 0.14],
                    'Sr': [87.62, 1042.15, 1655.0, 2.5, 50.0, 0.301],
                    'Si': [28.086, 1683.15, 3538.0, 1.5, 14.0, 0.705],
                    'Cu': [63.546, 1357.75, 2835.0, 1.6, 35.0, 0.385],
                    'H': [1.007, 14.175, 20.28, 0.79, 0.0, 14.304],
                    'Zn': [65.38, 692.88, 1180.0, 1.5, 35.0, 0.388],
                    'F': [18.998, 53.63, 85.03, 0.57, 10.0, 0.824],
                    'Ca': [40.078, 1112.15, 1757.0, 2.2, 20.0, 0.647],
                    'Se': [78.96, 494.15, 958.0, 1.2, 45.0, 0.321],
                    'Br': [79.904, 266.05, 332.0, 1.1, 45.0, 0.474],
                    'Mn': [54.938, 1519.15, 2334.0, 1.8, 30.0, 0.479],
                    'B': [10.8110, 2573.150, 4200.0, 1.20, 6.0, 1.026]}
        return atom_values[atom.GetSymbol()]

    def mean_interatomic_distance(self, atom, molecule):
        mean_distance = molecule.xyz.get_all_distances()[atom.GetIdx()].mean()
        return [mean_distance]

    def atom_soap(self, atom, molecule):
        return list(molecule.SOAP[atom.GetIdx()])


class Bond_Continuous_Featurizer(Continuous_Featurizer):
    """
    This class is used to extract continuous features from bonds in a molecule.

    Inherits from Continuous_Featurizer.

    Methods:
        bond_length(bond, molecule): Calculates the length of a bond in a molecule.
    """
    def __init__(self, continuous_features):
        super().__init__(continuous_features)

    def bond_length(self, bond, molecule):
        '''
        Returns the bond length for a single bond
        '''
        BeginAtomId=bond.GetBeginAtomIdx()
        EndAtomId=bond.GetEndAtomIdx()
        return [molecule.xyz.get_distance(BeginAtomId,EndAtomId)]


class MoleculeGraph():
    """
    This class creates a molecular graph with various properties and features and contains methods for calculating both categorical and continuous features of a molecule, as well as handling the molecule's structural information.

    Attributes:
        name (str): Name of the molecule.
        smiles (str): SMILES representation of the molecule.
        y (float): Target property of the molecule.
        index (int): Index of the molecule in a dataset.
        desc_names (list): List of RDKit descriptor names.
        feature_sets (dict): Dictionary specifying the feature sets to be used.
        other_molecule_features (list): Additional molecule features.
        structure_format (str): Format of the molecule structure file.
        soap_parameters (str): Parameters for SOAP feature calculation.
        datapath (str): Path to the data directory.

    Methods:
        get_categorical_node_features(): Returns categorical features of the molecule's nodes.
        get_continuous_node_features(): Returns continuous features of the molecule's nodes.
        get_soap_node_features(): Returns SOAP features of the molecule's nodes.
        get_rdkit_molecule_features(missingVal): Returns RDKit calculated molecular features.
        get_other_molecule_features(): Returns other specified molecular features, e.g., dG, that isn't calculated by RDKit but set manually.
        get_categorical_edge_features(): Returns categorical features of the molecule's edges.
        get_continuous_edge_features(): Returns continuous features of the molecule's edges.
        get_features(molpart): Retrieves features for a specified part of the molecule.
        get_xyz(): Obtains the XYZ coordinates of the molecule.
        get_edge_list(): Generates a list of edges in the molecular graph.
    """

    def __init__(self,
                 name,
                 smiles,
                 y,
                 index,
                 desc_names,
                 feature_sets,
                 other_molecule_features=[],
                 structure_format="xyz",
                 soap_parameters={"r_cut": 3.0, 
                                  "l_max": 4,
                                  "n_max": 4,
                                  "atom_sigma": 0.15,
                                  "compression_mode": "mu1nu1", 
                                  "species": [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53],},
                 datapath="../data/",):
        super(MoleculeGraph,self).__init__()

        self.name=name
        self.index=index
        self.smiles=smiles
        self.y=y
        self.soap_parameters=soap_parameters
        self.desc_names=desc_names
        self.structure_format=structure_format
        self.datapath=datapath
        self.molecule=Chem.AddHs(Chem.MolFromSmiles(smiles))
        self.edge_list=self.get_edge_list()
        self.xyz=self.get_xyz()
        self.other_molecule_features=other_molecule_features
        self.feature_sets=feature_sets
        self.molecule_features=self.get_features("molecule")
        self.node_features=np.hstack(self.get_features("node"))
        self.edge_features=np.hstack(self.get_features("edge"))

    def get_categorical_node_features(self):
        features=[atom_cat_featurizer.encode(atom) for atom in self.molecule.GetAtoms()]
        return np.array(features)

    def get_continuous_node_features(self):
        features=[atom_cont_featurizer.encode(atom,self) for atom in self.molecule.GetAtoms()]
        return np.array(features)

    def get_soap_node_features(self):
        if self.soap_parameters is None:
            return []
        elif self.soap_parameters is not None:
            molsoap = compute_SOAP(self.xyz,
                                   [self.soap_parameters["r_cut"], 
                                    self.soap_parameters["l_max"],
                                    self.soap_parameters["n_max"], 
                                    self.soap_parameters["atom_sigma"]],
                                   self.soap_parameters["compression_mode"],
                                   self.soap_parameters["species"])            
            return molsoap
    
    def get_rdkit_molecule_features(self, missingVal=None):
        from rdkit.Chem import Descriptors
        import traceback
        features=[]
        for desc in self.desc_names:
            try:
                val=getattr(Descriptors, desc)(self.molecule)
            except:
                traceback.print_exc()
                val=missingVal
            features.append(val)
        return features

    def get_other_molecule_features(self):
        # if self.other_molecule_features is None:
        #     return []
        # elif self.other_molecule_features is not None:
        return self.other_molecule_features

    def get_categorical_edge_features(self):
        features=[]
        for atom in self.molecule.GetAtoms():
            for bond in atom.GetBonds():
                features.append(bond_cat_featurizer.encode(bond))
        return np.array(features)

    def get_continuous_edge_features(self):
        features=[]
        for atom in self.molecule.GetAtoms():
            for bond in atom.GetBonds():
                features.append(bond_cont_featurizer.encode(bond,self))
        return np.array(features)

    def get_features(self,molpart):
        features=[]
        if len(self.feature_sets[molpart])>0:
            for type_ in self.feature_sets[molpart]:
                feature=getattr(self,"get_"+type_+"_"+molpart+"_features")()
                features.append(feature)
            return features
        elif len(self.feature_sets[molpart])==0:
            return None
        
    def get_xyz(self):
        if self.structure_format=="xyx" and "xyzs" in globals():
            return xyzs[self.name]
        else:
            from ase.io import read
            return read(self.datapath+"{}/{}.{}".format(self.structure_format,
                                               self.name,
                                               self.structure_format))
        
    def get_edge_list(self):
        atom_source=[]
        atom_target=[]
        for atom in self.molecule.GetAtoms():
            for bond in atom.GetBonds():
                atom_source.append(atom.GetIdx())
                if bond.GetEndAtomIdx()==atom.GetIdx():
                    atom_target.append(bond.GetBeginAtomIdx())
                else:
                    atom_target.append(bond.GetEndAtomIdx())
                if atom_source[-1]==atom_target[-1]:
                    print("Error: an edge cannot have the same begin and end index!")
                    break
            if atom_source[-1]==atom_target[-1]:
                break
        return torch.stack([torch.tensor(atom_source),torch.tensor(atom_target)])
        
        
class SKFGraphDataset():
    """
    This class creates dataset object suitable for stratified cross-validation in PyTorch through converting a list of MoleculeGraph objects into a list of PyTorch Data objects and manages the creation of training and validation/testing DataLoaders for each fold of cross-validation.

    Attributes:
        graphs (list): List of MoleculeGraph objects.
        skf_params (dict): Parameters for stratified k-fold cross-validation.

    Methods:
        set_torch_skfdataset(fold): Prepares train and test datasets for a specified fold.
        dataloaders(fold, generator, seed_worker): Returns DataLoaders for training and testing for a specified fold.
    """

    def __init__(self,
                 graphs,
                 skf_params,):
        super(SKFGraphDataset,self).__init__()

        self.graphs=graphs
        for k in skf_params.keys():
            self.__setattr__(k,skf_params[k])

    def set_torch_skfdataset(self, fold):
        train_index,test_index=self.skf_ids[fold]
        y_train = [self.graphs[i].y for i in train_index]
        train_dataset=[]
        test_dataset=[]
        for mg in self.graphs:
            x=mg.node_features
            edge_features=mg.edge_features
            datapoint=Data(x=torch.tensor(x,dtype=torch.float),
                           y=torch.tensor(mg.y, dtype=torch.float),
                           name=mg.name,
                           edge_index=mg.edge_list,
                           edge_attr=torch.tensor(edge_features, dtype=torch.float),
                           num_nodes=x.shape[0],
                           index=mg.index)
            if mg.molecule_features is not None:
                mol_features=np.hstack(mg.molecule_features)
                datapoint.__setattr__("molecule_features",
                                      torch.tensor(mol_features.reshape(1,-1),dtype=torch.float))
            if mg.index in train_index:
                train_dataset.append(datapoint)
            elif mg.index in test_index:
                test_dataset.append(datapoint)

        self.__setattr__("train_dataset",train_dataset)
        self.__setattr__("test_dataset",test_dataset)

        return train_dataset, test_dataset

    def dataloaders(self,fold,generator,seed_worker):
        train_dataset, test_dataset = self.set_torch_skfdataset(fold)
        train_loader = DataLoader(train_dataset,
                                    batch_size=self.batch_size,
                                    worker_init_fn=seed_worker,
                                    generator=generator,
                                    shuffle=True,)
        test_loader = DataLoader(test_dataset,
                                batch_size=self.batch_size,
                                worker_init_fn=seed_worker,
                                generator=generator,
                                shuffle=False,)
        return train_loader,test_loader


def compute_SOAP(xyz,
                 soap_params, 
                 compression_mode = "mu1nu1", 
                 species = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53],
                 periodic = False,):
    """
    This function calculates SOAP (Smooth Overlap of Atomic Positions) features for a molecule.

    Args:
        xyz (np.array): XYZ coordinates of the molecule.
        soap_params (str): List of parameters for SOAP feature calculation.
        species (list): List of atomic numbers of atoms to be considered as neighbors.
        compression (str): Compression mode for SOAP features.

    Returns:
        np.array: SOAP features of the molecule.
    """
    r_cut,l_max,n_max,sigma = soap_params
    compression={'mode': compression_mode}

    soap = descriptors.SOAP(species=species,
                            periodic=periodic,
                            r_cut=r_cut,
                            n_max=int(n_max),
                            l_max=int(l_max),
                            sigma=sigma,
                            compression=compression,
                            )
    return soap.create(xyz)

    
def update_soaps(molgraphs, 
                 params,
                 compression_mode = "mu1nu1",
                 species = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53],
                 verbose = False):
    '''
    For use when the only node feature are SOAPs. Updates the node 
    features with SOAPs calculated with a different parameterization 
    while leaving the rest of the molecular graph unchanged. 
    '''
    r_cut,l_max,n_max,sigma = params
    compression={'mode': compression_mode}

    soap = descriptors.SOAP(species=species,
                            periodic=False,
                            r_cut=r_cut,
                            n_max=int(n_max),
                            l_max=int(l_max),
                            sigma=sigma,
                            compression=compression,
                            )
    updated_molgraphs = []
    for i,mg in enumerate(molgraphs):
        mg.node_features=soap.create(mg.xyz)
        updated_molgraphs.append(mg)
        if verbose and (i+1)%500==0:
            print("Created SOAP {}".format(i+1))
    return updated_molgraphs


# Todo: turn this into a function
# Get atomic species/symbols in the dataset 
# symbols=set()
# for smi in smiles:
#     mol = molecule_from_smiles(smi)
#     symbols.update([atom.GetSymbol() for atom in mol.GetAtoms()])


# Load periodic table data, change path if necessary
file = "../data/periodic.csv" 
periodic_data=pd.read_csv(file)

    
# Need to update here the atomic species that are present in the dataset
symbols = {'I', 'Cl', 'Br', 'S', 'N', 'F', 'B', 'C', 'P', 'O', 'H'}


# The below sets up the atom and bond featurizers that will be used. These can be changed as needed by commenting out the features that are not needed or adding new features in each of the dictionaries. 

atom_cat_features = {"symbol": symbols,
                    "n_valence": {0, 1, 2, 3, 4, 5, 6},
                    "n_hydrogens": {0, 1, 2, 3, 4},
                    "hybridization": {"s", "sp", "sp2", "sp3"},
                    "isin_ring":{True, False},
                    "nodal_degree":{1,2,3,4,5},
                    "chiral_tag":{'CHI_TETRAHEDRAL_CCW', 'CHI_TETRAHEDRAL_CW', 'CHI_UNSPECIFIED'},
                    "charge":{-1,  0,  1,  2},
                    "aromatic":{True, False},
                    "radical_electrons":{0,1},
                    "number_bonds":{0,1,2,3,4},
                    "atom_types":{'Alkali Metal',
                                  'Alkaline Earth Metal',
                                  'Halogen', 'Metal',
                                  'Metalloid',
                                  'Nonmetal',
                                  'Transition Metal'},}
atom_cat_featurizer = Atom_Categorical_Featurizer(atom_cat_features)


bond_cat_features = {"bond_type": {"single", "double", "triple", "aromatic"},
                    "conjugated": {True, False},
                    "isin_ring": {True, False},
                    "bond_has_H": {True,False},}
bond_cat_featurizer = Bond_Categorical_Featurizer(bond_cat_features)


bond_cont_features = ["bond_length"]
bond_cont_featurizer = Bond_Continuous_Featurizer(bond_cont_features)


atom_cont_features = ["element_data"]
atom_cont_featurizer = Atom_Continuous_Featurizer(atom_cont_features)
