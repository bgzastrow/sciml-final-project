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

import scipy as sp
import numpy            as np
import torch
from torch.utils.data   import Dataset, DataLoader
# import src.mace.utils   as utils
# from pathlib import Path
# from scipy.integrate import odeint, solve_ivp

### ----------------------- 1D L96 models ----------------------- ###


def dynamics(t, x, F):
    """Lorenz 96 model with constant forcing (F)."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def run_dynamics(
        times,
        state_dimension,
        forcing_magnitude,
        perturbation_scale=0.01,
        seed=42,
        ):
    """Generate a training snapshot matrix."""

    if state_dimension < 4:
        raise ValueError("state_dimension must be >= 4 for Lorenz 96.")

    # Initial state (equilibrium)
    initial_state = forcing_magnitude * np.ones(state_dimension)

    # Perturb system
    rng = np.random.default_rng(seed=seed)
    perturbation_index = rng.integers(0, state_dimension)
    perturbation = perturbation_scale * forcing_magnitude
    initial_state[perturbation_index] += perturbation

    # Integrate dynamics
    result = sp.integrate.solve_ivp(
        dynamics,
        t_span=(times[0], times[-1]),
        y0=initial_state,
        method="RK45",
        t_eval=times,
        args=(forcing_magnitude,),
        rtol=1e-6,
        atol=1e-9,
    )
    states = result.y.T

    return states


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
            2. Select a random test force, that is not in the training set 
                --> self.testF
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
        self.nb_test = nb_test
        self.n_L96 = n_L96
        self.fraction = fraction
        self.train = train

        # Forcing levels
        self.F_min = 1.0
        self.F_max = 2.0
        F_vals = np.linspace(self.F_min, self.F_max, nb_samples)

        # Split into train and test set        
        N = int(self.fraction*len(F_vals))
        seed = 42 # make this random but repeatable
        rng = np.random.default_rng(seed)
        F_select = rng.choice(len(F_vals), size=len(F_vals), replace=False)

        # Get a random, non-ordered choice of F values in the train and test datasets
        if self.train:
            self.F_vals = F_vals[F_select[:N]]
        else:
            self.F_vals = F_vals[F_select[N:]]

        # Select a random test force, that is not in the training set
        F_idx_test = rng.choice(np.arange(N, len(F_vals)), size=nb_test)
        self.testF = F_vals[F_select[F_idx_test]]
            
    def __len__(self):
        '''
        Return the length of the dataset (number of 1D models used for training or validation).
        '''
        return len(self.F_vals)

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
        # THIS STEP RUNS THE L96 FOM
        mod = L96mod(self.F_vals[idx], self.n_L96)

        dt, n, p = mod.split_in_0D()

        return torch.from_numpy(n), torch.from_numpy(p), torch.from_numpy(dt)
    

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
        T_max = 10 # 10 # maximum time for simulation
        dt = 0.0001 # 0.001 # timestep
        self.time = np.arange(0.0, T_max, dt)
        self.n = run_dynamics(self.time, n_L96, F)
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
        p    = self.force*np.ones(len(self.time)-1).T
        return dt.astype(np.float64), n_0D.astype(np.float64), p.T.astype(np.float64)


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
    # Make PyTorch dataset
    train = L96data(
        nb_samples=nb_samples,
        n_L96=n_L96,
        dt_fract=dt_fract,
        nb_test=nb_test,
        train=True,
        )
    valid = L96data(
        nb_samples=nb_samples,
        n_L96=n_L96,
        dt_fract=dt_fract,
        nb_test=nb_samples,
        train=False,
        )
    
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True ,  **kwargs)
    valid_loader = DataLoader(dataset=valid, batch_size=1 , shuffle=False,  **kwargs)

    return train, valid, train_loader, valid_loader


def get_test_data(testF, n_L96):
    '''
    Get the data of the test 1D model, given a path and meta-data from a training setup.

    Similar procedure as in the __getitem__() of the L96data class.

    The specifics of the 1D test model are stored in the 'name' dictionary.

    Input:
        - testF [str]: forcing value of the test model
    '''
    
    mod = L96mod(testF, n_L96)
    dt, n, p = mod.split_in_0D()
    name = {'F' : mod.force}

    return mod, (torch.from_numpy(n), torch.from_numpy(p), torch.from_numpy(dt)), name

