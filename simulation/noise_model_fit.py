import numpy as np
import healpy as hp
from scipy.optimize import curve_fit
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed


import numpy as np

class MCfitting:
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def noise_model(self, ell, *params):
        ell_knee, sigma, beta = params
        n_ell = sigma * (1 + (ell_knee / ell)) ** beta
        return n_ell
    
    def polynomial_noise_model(self, ell, *params):
        a, b, c, d, e, f, g, h, i, j, k = params
        return a + b*ell + c*ell**2 + d*ell**3 + e*(1/ell) + f*(1/ell)**2 + g*(1/ell)**3 + h*ell**4 + i*(1/ell)**4 + j*(1/ell)**5 + k*ell**5

    def likelihood_func(self, noise_params, data_std, model='noise'):
        if model == 'noise':
            model_y = self.noise_model(self.data_x, *noise_params)
        elif model == 'polynomial':
            model_y = self.polynomial_noise_model(self.data_x, *noise_params)
        else:
            raise ValueError("Model type not recognized. Use 'noise' or 'polynomial'.")
        
        loglike = (self.data_y - model_y) ** 2 / (2 * data_std ** 2)
        loglike = -np.sum(loglike)
        return loglike

    def mcmc_sampler(self, initial_params, data_std, steps=10000, step_vec=None,
                     convergence_threshold=1e-6, convergence_steps=1000, check_interval=100, burn_in=1000, model='noise'):
        params = np.array(initial_params)
        if step_vec is None:
            step_vec = np.ones(len(params)) * 0.1  # Default step vector with the same length as params
        else:
            step_vec = np.array(step_vec)
            if len(step_vec) != len(params):
                raise ValueError("step_vec must have the same length as initial_params")

        samples = [params]
        likelihoods = [self.likelihood_func(params, data_std, model)]
        stable_count = 0

        for step in range(steps):
            new_params = params + np.random.randn(len(params)) * step_vec
            new_likelihood = self.likelihood_func(new_params, data_std, model)
            old_likelihood = self.likelihood_func(params, data_std, model)

            if np.random.rand() < np.exp(new_likelihood - old_likelihood):
                params = new_params
                likelihoods.append(new_likelihood)
                samples.append(params)
            else:
                likelihoods.append(old_likelihood)
                samples.append(params)

            if step > 0 and step % check_interval == 0:
                recent_samples = np.array(samples[-check_interval:])
                param_change = np.mean(np.abs(np.diff(recent_samples, axis=0)), axis=0)
                change = np.mean(param_change)

                if change < convergence_threshold:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= convergence_steps:
                    print(f"Convergence reached after {step} steps.")
                    break

        return np.array(samples[burn_in:]), np.array(likelihoods[burn_in:])

class new_MCfitting:
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def noise_model(self, ell, *params):
        ell_knee, sigma, beta = params
        n_ell = sigma * (1 + (ell_knee / ell)) ** beta
        return n_ell

    def polynomial_noise_model(self, ell, *params):
        a, b, c, d, e, f, g, h, i, j, k = params
        return (a + b * ell + c * ell**2 + d * ell**3 + e * (1 / ell) + 
                f * (1 / ell)**2 + g * (1 / ell)**3 + h * ell**4 + 
                i * (1 / ell)**4 + j * (1 / ell)**5 + k * ell**5)

    def log_likelihood(self, noise_params, data_std, model='noise'):
        """ Compute the log-likelihood for the given model and parameters. """
        if model == 'noise':
            model_y = self.noise_model(self.data_x, *noise_params)
        elif model == 'polynomial':
            model_y = self.polynomial_noise_model(self.data_x, *noise_params)
        else:
            raise ValueError("Model type not recognized. Use 'noise' or 'polynomial'.")
        
        residuals = self.data_y - model_y
        loglike = -0.5 * np.sum((residuals / data_std) ** 2)
        return loglike

    def log_prior(self, params):
        """ Add priors to constrain parameter values. """
        if np.any(params < 0):  # Example prior: all parameters must be non-negative
            return -np.inf
        return 0.0  # Flat prior for now

    def log_posterior(self, params, data_std, model='noise'):
        """ Log posterior = log prior + log likelihood. """
        log_prior = self.log_prior(params)
        if not np.isfinite(log_prior):
            return -np.inf
        return log_prior + self.log_likelihood(params, data_std, model)

    def adaptive_step_size(self, acceptance_rate, step_vec):
        """ Adapt the step size to maintain an optimal acceptance rate (~0.25). """
        target_acceptance = 0.25
        if acceptance_rate < target_acceptance:
            step_vec *= 0.9  # Decrease step size if acceptance is too low
        elif acceptance_rate > target_acceptance:
            step_vec *= 1.1  # Increase step size if acceptance is too high
        return step_vec

    def mcmc_sampler(self, initial_params, data_std, steps=10000, step_vec=None, 
                     convergence_threshold=1e-6, convergence_steps=1000, 
                     check_interval=100, burn_in=1000, model='noise'):
        """ Perform MCMC sampling using an adaptive Metropolis-Hastings algorithm. """
        params = np.array(initial_params)
        n_params = len(params)

        if step_vec is None:
            step_vec = np.ones(n_params) * 0.1  # Default step vector
        else:
            step_vec = np.array(step_vec)
            if len(step_vec) != n_params:
                raise ValueError("step_vec must have the same length as initial_params")

        samples = [params]
        log_likes = [self.log_posterior(params, data_std, model)]
        acceptance_count = 0
        stable_count = 0

        for step in range(steps):
            # Propose new parameters
            new_params = params + np.random.randn(n_params) * step_vec
            new_loglike = self.log_posterior(new_params, data_std, model)
            old_loglike = self.log_posterior(params, data_std, model)

            # Acceptance criteria
            if np.random.rand() < np.exp(new_loglike - old_loglike):
                params = new_params
                log_likes.append(new_loglike)
                samples.append(params)
                acceptance_count += 1
            else:
                log_likes.append(old_loglike)
                samples.append(params)

            # Adapt step size based on acceptance rate
            if (step + 1) % check_interval == 0:
                acceptance_rate = acceptance_count / check_interval
                step_vec = self.adaptive_step_size(acceptance_rate, step_vec)
                acceptance_count = 0

                # Check for convergence
                recent_samples = np.array(samples[-check_interval:])
                param_change = np.mean(np.abs(np.diff(recent_samples, axis=0)), axis=0)
                change = np.mean(param_change)

                if change < convergence_threshold:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= convergence_steps:
                    print(f"Convergence reached after {step} steps.")
                    break

        return np.array(samples[burn_in:]), np.array(log_likes[burn_in:])

class curvefit:
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def noise_model(self, ell, ell_knee, sigma, beta):
        n_ell = sigma * (1 + (ell_knee / ell)) ** beta
        return n_ell
    
    def polynomial_noise_model(self, ell, a, b, c, d, e, f, g, h, i, j, k):
        return a + b*ell + c*ell**2 + d*ell**3 + e*(1/ell) + f*(1/ell)**2 + g*(1/ell)**3 + h*ell**4 + i*(1/ell)**4 + j*(1/ell)**5 + k*ell**5
    
    def run_fit(self, num_trial, model='noise'):
        if model == 'noise':
            params, cov = curve_fit(self.noise_model, self.data_x, self.data_y, maxfev=num_trial)
        elif model == 'polynomial':
            params, cov = curve_fit(self.polynomial_noise_model, self.data_x, self.data_y, maxfev=num_trial)
        else:
            raise ValueError("Model type not recognized. Use 'noise' or 'polynomial'.")
        return params, cov

class grid_search:
    def __init__(self, param_ranges, data_x, data_y, model = 'noise'):
        self.param_ranges = param_ranges
        self.data_x = data_x
        self.data_y = data_y
        self.model = model

    def noise_model(self, ell, ell_knee, sigma, beta):
        n_ell = sigma * (1 + (ell_knee / ell)) ** beta
        return n_ell
    
    def polynomial_noise_model(self, ell, a, b, c, d, e, f, g, h, i, j, k):
        return a + b*ell + c*ell**2 + d*ell**3 + e*(1/ell) + f*(1/ell)**2 + g*(1/ell)**3 + h*ell**4 + i*(1/ell)**4 + j*(1/ell)**5 + k*ell**5
    
    def objective_function(self, params):
        if self.model == 'noise':
            model_y = self.noise_model(self.data_x, *params)
        elif self.model == 'polynomial':
            model_y = self.polynomial_noise_model(self.data_x, *params)
        else:
            raise ValueError("Model type not recognized. Use 'noise' or 'polynomial'.")
        
        loglike = - np.sum(np.log((self.data_y - model_y) ** 2))
        return loglike
    
    def perform_grid_search(self, n_jobs=1):
        param_combinations = list(ParameterGrid(self.param_ranges))
        
        # Using parallel processing to speed up the grid search
        results = Parallel(n_jobs=n_jobs)(delayed(self.objective_function)(params) for params in param_combinations)
        
        best_mse = float('inf')
        best_params = None
        for i, mse in enumerate(results):
            if mse is not None and mse < best_mse:
                best_mse = mse
                best_params = param_combinations[i]

        return best_params, best_mse