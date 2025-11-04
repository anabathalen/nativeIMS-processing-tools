import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# gaussian fit function
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(- ((x - mean)**2) / (2 * stddev **2))

# r2 Calculation
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) **2)
    return 1 - (ss_res / ss_tot)

# fit a single gaussian to a dataset, retrying with different initial guesses
def fit_gaussian_with_retries(x_values, y_values, n_attempts = 10, random_state=None):
    rng = np.random.default_rng(random_state)
    best_r2 = -np.inf
    best_params = None
    best_fitted_values = None

    # looping the guessing and fitting process n times where n is the number of retries inputted
    for attempt in range(n_attempts):
        # picking a sensible set of starting values for the amplitude, mean and standard deviation of the gaussian fit as a starting point for fitting
        initial_guess = [
            max(y_values), # maximum y value
            rng.uniform(np.min(x_values), np.max(x_values)), # random value between the minimum and maximum x values
            rng.uniform(0.1 * np.std(x_values), 2 * np.std(x_values)) # random values between 0.1 and 2 x the standard deviation of the 
        ]

        try:
            params, _ = curve_fit(gaussian, x_values, y_values, p0 = initial_guess) # using scipy.optimize.curve_fit to fit a gaussian to the dataset
            fitted_values = gaussian(x_values, *params) # using the fitted parameters (stored in params) to generate fitted y values - the * unpacks params
            r2 = r_squared(y_values, fitted_values)

            if r2 > best_r2: # only accept the values if they are the best ones
                best_r2 = r2
                best_params = params
                best_fitted_values = fitted_values
        
        except RuntimeError:
            continue
    
    return best_params, best_r2, best_fitted_values