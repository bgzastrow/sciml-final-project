import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from torch.utils.data   import Dataset, DataLoader


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


def generate_data(
        times,
        state_dimension,
        forcing_magnitudes,
        ):
    """Generate a set of training snapshot matrices."""

    samples = []
    for forcing_magnitude in forcing_magnitudes:
        all_states = run_dynamics(
            times=times,
            state_dimension=state_dimension,
            forcing_magnitude=forcing_magnitude,
        )
        samples.append(all_states.T)


def save_data():
    pass


def load_data():
    pass

class L96Data(Dataset):
    """Lorenz 96 dataset."""

    def __init__(
            self,
            num_1d_training_samples,
            num_1d_testing_samples,
            state_dimension,
            training=True,
            training_fraction=0.7,
    ):
        self.num_1d_training_samples = num_1d_training_samples
        self.num_1d_testing_samples = num_1d_testing_samples
        self.state_dimension = state_dimension
        self.training_fraction = training_fraction

        # Populate a list of forcing magnitudes
        min_forcing_magnitude = 1.0
        max_forcing_magnitude = 2.0
        forcing_magnitudes = np.linspace(min_forcing_magnitude, max_forcing_magnitude, num_1d_testing_samples)

        # Select a list of forcing magnitudes for training or testing
        seed = 42
        rng = np.random.default_rng(seed=seed)
        forcing_indexes = rng.choice(
            num_1d_testing_samples,
            size=num_1d_testing_samples,
            replace=False,
            )
        num_1d_samples = int(self.training_fraction * num_1d_testing_samples)
        if training:
            self.forcing_magnitudes = forcing_magnitudes[forcing_indexes[:num_1d_samples]]
        else:
            self.forcing_magnitudes = forcing_magnitudes[forcing_indexes[num_1d_samples:]]

    def __len__(self):
        return len(self.forcing_magnitudes)
    
    def __getitem__(self, idx):
        return torch.samples_0d, self.forcing_magnitudes