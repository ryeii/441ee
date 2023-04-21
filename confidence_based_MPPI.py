import numpy as np
import pandas as pd

'''
Convention:
- x (numpy array): zone state + environment state, shape (state_dim,)
    - x[0]: zone temperature
    - x[1]: outdoor air drybulb temperature
    - x[2]: outdoor air relative humidity
    - x[3]: avg. site wind speed
    - x[4]: avg. site total solar gain
    - x[5]: zone occupant count
- u (float): the control signal, int in [0, 9], shape (control_dim,)
- t (int): the current time
'''


COMFORT_RANGE = (23, 26)
WEATHER = pd.read_csv('weather.csv')
LAMBDA_CONFIDENCE = 1.0

model = 0 #TODO load model


def env_reader(timestep):
    '''
    Return the environment state at the given timestep.

    Args:
    - timestep (int): timestep to read the environment state from
    - weather_df (pandas dataframe): the weather dataframe

    Returns:
    - env_state (numpy array): the environment state at the given timestep, shape (1,)
    '''
    return WEATHER.iloc[timestep, 1:].values


def get_confidence_value(var):
    '''
    Return the confidence value given the variance.

    Args:
    - var (float): the variance of the Gaussian Process prediction

    Returns:
    - confidence_value (float): the confidence value
    '''
    return 0.5 * np.log(2 * np.pi * var) + 0.5 # information entropy of Gaussian is 1/2 * log(2*pi*var) + 1/2


def dynamics(x, u, t):
    '''
    Predict the next zone temperature given the current state, control signal, and time.

    Args:
    - x (float): zone state (zone temperature), shape (state_dim,)
    - u (float): the control signal, shape (control_dim,)
    - t (float): the current time

    Returns:
    - mu (float): the mean of the Gaussian Process prediction
    - var (float): the variance of the Gaussian Process prediction
    '''
    model.eval()

    env_state = env_reader(t)
    x = np.concatenate((x, env_state)) # concatenate the zone state and environment state
    x = np.concatenate((x, np.array([u]))) # concatenate state and the control signal

    pred = model(x, u)

    mu = pred.mean
    var = pred.variance

    return mu, var

def cost(x, var, u):
    '''
    Compute the cost of the given state and control signal.

    Args:
    - x (numpy array): zone state + environment state, shape (state_dim,)
    - sigma (float): standard deviation of the Gaussian Process prediction
    - u (float): the control signal, shape (control_dim,)

    Returns:
    - c (float): the cost of the given state and control signal
    '''
    comfort_range = COMFORT_RANGE
    weight_energy = 1
    if x[5] > 0:
        weight_energy = 0.1

    comfort_cost = -(1 - weight_energy) * (abs(x[0] - comfort_range[0]) + abs(x[0] - comfort_range[1]))
    energy_cost = weight_energy * (u - x[0])**2

    confidence = get_confidence_value(var)

    return comfort_cost + energy_cost - LAMBDA_CONFIDENCE * confidence

class MPPIController:
    def __init__(self, num_samples, horizon, time_offset, dynamics_fn, cost_fn, lambda_=1.0, sigma=1.0):
        self.num_samples = num_samples
        self.horizon = horizon
        self.time_offset = time_offset
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.lambda_ = lambda_
        self.sigma = sigma
        
    def control(self, x0, t):
        """
        Compute the MPPI control signal given the current state and time.

        Args:
        - x0 (numpy array): the current state, shape (state_dim,)
        - t (float): the current time

        Returns:
        - u (numpy array): the control signal, shape (control_dim,)
        """
        S = np.zeros((self.num_samples, self.horizon)) # sample trajectories
        C = np.zeros((self.num_samples,)) # trajectory costs
        U = np.zeros((self.horizon,)) # control signal

        # populate U with random signals from 0 to 1
        for j in range(self.horizon):
            U[j] = np.random.uniform(0, 1)
        
        for i in range(self.num_samples):
            x = np.copy(x0)
            s = np.random.normal(0, self.sigma, (self.horizon,)) # sample noise
            for j in range(self.horizon):
                u = U[j] + s[j]
                x, var = self.dynamics_fn(x, u, self.time_offset + t) # pass t so it can call env_reader to get weather
                S[i, j] = x
                C[i] += self.cost_fn(x, var, u) # occupancy is obtained from the state, so we don't need to pass t
        
        expC = np.exp(-self.lambda_ * C)
        expC /= np.sum(expC)
        
        for j in range(self.horizon):
            U[j] = np.sum(expC * S[:, j])
        
        return U
