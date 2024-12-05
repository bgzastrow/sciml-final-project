'''
This script contains classes for loading the Lorenz 96 models,
which is the training data for MACE,
as well as code to preprocess the data.

Contains:
    Classes:
        - L96data: Dataset class (PyTorch) to prepare the dataset for training and validating the emulator
        - L96mod: Class to load a 1D L96 model
    
    Functions:
        - get_data: prepare the data for training and validating the emulator, using the PyTorch-specific dataloader
        - get_test_data: get the data of the test a 1D model, given a path and meta-data from a training setup
        - get_abs: get the abundances, given the normalised abundances
        - get_phys: reverse the normalisation of the physical parameters
        - read_input_1Dmodel: read input text file of 1D L96 models, given the filename
        - read_data_1Dmodel: read data text file of output abundances of 1D L96 models

NOTE:
This script only works for this specific dataset.

'''

import numpy            as np
import torch
from torch.utils.data   import Dataset, DataLoader
import src.mace.utils   as utils
from pathlib import Path

specs_dict, idx_specs = utils.get_specs()

### ----------------------- 1D L96 models ----------------------- ###

def run_L96(F, t_arr, n_L96):
    """ Generate an L96 dataset for a given forcing constant and time t 
        See https://en.wikipedia.org/wiki/Lorenz_96_model
    """
    def L96(x, t):
        """Lorenz 96 model with constant forcing"""
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    
    x0 = F * np.ones(n_L96)  # Initial state (equilibrium)
    rand_int = np.random.randint(0, high=n_L96) # perturb a random state
    x0[rand_int] += F/100  # Add small perturbation from equilibrium
    x = odeint(L96, x0, t_arr)
    return x

class L96data(Dataset):
    '''
    Class to initialise the dataset to train & test emulator.

    More specifically, this Dataset uses 1D L96 models, and splits them in 0D models.
    '''
    def __init__(self, nb_samples, n_L96, dt_fract, nb_test, train=True, fraction=0.7, cutoff = 1e-20, scale = 'norm'):
        '''
        Initialising the attributes of the dataset.

        Input:
            - nb_samples [int]: number of 1D models to use for training & validation
            - n_L96 [int]: dimension of L96 model to be used for training & validation
            - dt_fract [float]: fraction of the timestep to use
            - nb_test [int]: number of models to uses for testing
            - train [boolean]: True if training, False if testing
            - fraction [float]: fraction of the dataset to use for training, 1-fraction is used for validation,
                default = 0.7
            - cutoff [float]: cutoff value for the L96 variables,
                default = 1e-20
            - scale [str]: type of scaling to use, default = 'norm'

        Preprocess on data:
            - clip all variables to cutoff
            - take np.log10 of variables

        Structure:
            1. Load the paths of the 1D models
            2. Select a random test path, that is not in the training set 
                --> self.testpath
            3. Set the min and max values of the variables 
                resulting from a search through the full dataset
                These values are used for normalisation of the data. 
                --> self.mins, self.maxs
            4. Set the cutoff value for the variables 
                --> self.cutoff
            5. Set the fraction of the dataset to use for training 
                --> self.fraction
            6. Split the dataset in train and test set 
        '''

        self.idxs = utils.generate_random_numbers(nb_samples, 0, len(paths))
        self.path = paths[self.idxs]

        ## select a random test path, that is not in the training set
        # self.test_idx = utils.generate_random_numbers(1, 0, len(paths))
        self.testpath = list()
        # self.testpath.append(paths[self.test_idx][0])
        self.nb_test = nb_test
        count = 0

        ## These values are the results from a search through the full dataset; see 'minmax.json' file
        Fmin_linear = 0.001
        Fmax_linear = 1e6
        self.F_min = np.log10(Fmin_linear)
        self.F_max = np.log10(Fmax_linear)
        F_vals = np.logspace(Fmin_linear, Fmax_linear, nb_samples)
        self.dt_max = 434800000000.0
        self.dt_fract = dt_fract
        self.n_min = np.log10(cutoff)
        self.n_max = np.log10(0.85e-1)    

        self.mins = np.array([self.F_min, self.n_min, self.dt_fract])
        self.maxs = np.array([self.F_max, self.n_max, self.dt_max])

        self.cutoff = cutoff
        self.fraction = fraction
        self.train = train

        ## Split in train and test set        
        N = int(self.fraction*len(self.path))
        seed=42 # make this random but repeatable
        rng = np.random.default_rng(seed)
        F_select = rng.choice(len(F_vals), size=len(F_vals), replace=False)
        # get a random, non-ordered choice of F values in the train and test datasets
        if self.train:
            self.F_vals = F_vals[F_select[:N]]
        else:
            self.F_vals = F_vals[F_select[N:]]
            
    def __len__(self):
        '''
        Return the length of the dataset (number of 1D models used for training or validation).
        '''
        return len(self.path)

    def __getitem__(self, idx):
        '''
        Get the data of the idx-th 1D model.

        The L96mod class is used to get the data of the 1D model.
        Subsequently, this data is preprocessed:
            - L96 variables (n) are
                - clipped to the cutoff value
                - np.log10 is taken 
                - normalised to [0,1]
            - physical parameters (p) are
                - np.log10 is taken
                - normalised to [0,1]
            - timesteps (dt) are 
                - scaled to [0,1]
                - multiplied with dt_fract

        Returns the preprocessed data in torch tensors.
        '''
        # print(len(self))
        # print(idx)
        mod = L96mod(self.F_vals[idx])

        dt, n, p = mod.split_in_0D()

        ## physical parameters
        p_transf = np.empty_like(p)
        for j in range(p.shape[1]):
            p_transf[:,j] = utils.normalise(np.log10(p[:,j]), self.mins[j], self.maxs[j])

        ## transform L96 variables
        n_transf = np.clip(n, self.cutoff, None)
        n_transf = np.log10(n_transf)
        n_transf = utils.normalise(n_transf, self.n_min, self.n_max)

        ## timesteps
        dt_transf = dt/self.dt_max * self.dt_fract             ## scale to [0,1] and multiply with dt_fract

        return torch.from_numpy(n_transf), torch.from_numpy(p_transf), torch.from_numpy(dt_transf)
    

def get_data(nb_samples, n_L96, dt_fract, nb_test, batch_size, kwargs):
    '''
    Prepare the data for training and validating the emulator.

    1. Make PyTorch dataset for the training and validation set.
    2. Make PyTorch dataloader for the 
        training 
            - batch size = batch_size
            - shuffle = True
        and validation set.
            - batch size = 1
            - shuffle = False 

    kwargs = {'num_workers': 1, 'pin_memory': True} for the DataLoader        
    '''
    ## Make PyTorch dataset
    train = L96data(nb_samples=nb_samples, n_L96=n_L96, dt_fract=dt_fract, nb_test = nb_test   , train = True)
    valid = L96data(nb_samples=nb_samples, n_L96=n_L96, dt_fract=dt_fract, nb_test = nb_samples, train = False)
    
    print('Dataset:')
    print('------------------------------')
    print('  total # of samples:',len(train)+len(valid))
    print('#   training samples:',len(train))
    print('# validation samples:',len(valid) )
    print('               ratio:',np.round(len(valid)/(len(train)+len(valid)),2))
    print('     #  test samples:',train.nb_test)

    data_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True ,  **kwargs)
    test_loader = DataLoader(dataset=valid, batch_size=1 , shuffle=False,  **kwargs)

    return train, valid, data_loader, test_loader


def get_test_data(testpath, n_L96, meta, inpackage = False):
    '''
    Get the data of the test 1D model, given a path and meta-data from a training setup.

    Similar procedure as in the __getitem__() of the L96data class.

    The specifics of the 1D test model are stored in the 'name' dictionary.

    Input:
        - testpath [str]: path of the 1D test model
        - meta [dict]: meta data from the training setup
    '''
    
    data = L96data(nb_samples=meta['nb_samples'], n_L96=n_L96, dt_fract=meta['dt_fract'],nb_test= 100, train=True, fraction=0.7, cutoff = 1e-20, scale = 'norm')
    mod = L96mod(testpath, inpackage)
    dt, n, p = mod.split_in_0D()

    name = {'F' : data.force}

    ## physical parameters
    p_transf = np.empty_like(p)
    for j in range(p.shape[1]):
        p_transf[:,j] = utils.normalise(np.log10(p[:,j]), data.mins[j], data.maxs[j])

    ## L96 variables
    n_transf = np.clip(n, data.cutoff, None)
    n_transf = np.log10(n_transf)
    n_transf = utils.normalise(n_transf, data.n_min, data.n_max)

    ## timesteps
    dt_transf = dt/data.dt_max * data.dt_fract             ## scale to [0,1] and multiply with dt_fract

    return mod, (torch.from_numpy(n_transf), torch.from_numpy(p_transf), torch.from_numpy(dt_transf)), name


def get_abs(n):
    '''
    Get the L96 variables, given the normalized variables.

    This function reverses the normalization of the L96 variables.
    '''
    cutoff = 1e-20
    nmin = np.log10(cutoff)
    nmax = np.log10(0.85e-1)

    return 10**utils.unscale(n,nmin, nmax)

def get_phys(p_transf,dataset):
    '''
    Reverse the normalization of the physical parameters.
    '''
    p = torch.empty_like(p_transf)
    for j in range(p_transf.shape[1]):
        p[:,j] = 10**utils.unscale(p_transf[:,j],dataset.mins[j], dataset.maxs[j])
    
    return p

class L96mod():
    '''
    Class to load the L96 model.
    '''
    def __init__(self, F, n_L96):
        '''
        Calculate the L96 model for a given number of degrees of freedom.
            - the L96 variables            --> self.n
            - the forcing constant         --> self.force
            - the time steps               --> self.time
        '''
        # should T_max and dt be adjusted?? these could have effects on success
        T_max = 100 # maximum time for simulation
        dt = 0.1 # timestep
        self.time = np.arange(0.0, T_max, dt)
        self.n = run_L96(F, self.time, n_L96)
        self.force = F

    def __len__(self):
        '''
        Return the length of the time array, which indicates the length of the 1D model.
        '''
        return len(self.time)

    def get_time(self):
        '''
        Return the time array of the model.
        '''
        return self.time
    
    def get_phys(self):
        '''
        Return the physical parameters of the model.
        '''
        return self.force
    
    def get_abs(self):
        '''
        Return the L96 variables.
        '''
        return self.n
    
    def get_dt(self):
        '''
        Return the time steps of the 1D model.
        '''
        return self.time[1:] - self.time[:-1]
    
    def split_in_0D(self):
        '''
        Split the 1D model in 0D models.
        '''
        dt   = self.get_dt()
        n_0D = self.get_abs()
        p    = np.array(self.force)
        return dt.astype(np.float64), n_0D.astype(np.float64), p.T.astype(np.float64)
