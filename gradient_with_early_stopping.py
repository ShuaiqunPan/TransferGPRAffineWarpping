#!/usr/bin/env python
# coding: utf-8
'''
Script Name: gradient.py
Description: This script is the program of implementing the cost function, riemannian gradient descent method.
'''
import math
import numpy as np
from scipy.linalg import expm, logm
from utils import *
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace.conditions import InCondition
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from numpy_ml.neural_nets.schedulers import ConstantScheduler, ExponentialScheduler, KingScheduler
from scipy.stats import special_ortho_group
import time
from scipy.special import psi, betaln
from scipy.integrate import quad
from scipy.stats import beta as beta_dist
from scipy.integrate import quad_vec
from numba import jit
import warnings
from scipy.integrate import quadrature
import scipy.special as sp


def cost_function(x, y, alphas, betas, weight, translation, m, dimension):
    '''
    Implement the cost function that includes rotation, translation, and Beta CDF transformation.
    '''
    n_T = x.shape[0]

    # Define the bound for the source and target problems.
    problem1_lower_bound = np.full(dimension, -5., dtype=np.float64)
    problem1_upper_bound = np.full(dimension, 5., dtype=np.float64)

    # (1) `x` is assumed to be in [-5, 5], no affine transformation yet

    # (2) Scale x to [0,1]
    x_scaled = (x - problem1_lower_bound) / (problem1_upper_bound - problem1_lower_bound)
    x_scaled = x_scaled.astype(np.float64)

    # (3) Apply the Beta CDF transformation
    if dimension == 1:
        x_scaled = x_scaled.flatten() # ensure 1D
        x_beta_transformed = beta_dist.cdf(x_scaled, float(alphas[0]), float(betas[0]))
    else:
        x_beta_transformed = np.zeros_like(x_scaled, dtype=np.float64)
        for i in range(dimension):
            x_beta_transformed[:, i] = beta_dist.cdf(x_scaled[:, i], float(alphas[i]), float(betas[i]))
    
    # (4) Rescale from [0,1] back to [-5,5]
    x_rescaled = x_beta_transformed * (problem1_upper_bound - problem1_lower_bound) + problem1_lower_bound

    if x_rescaled.ndim == 1:
        x_rescaled = x_rescaled.reshape(-1, 1)

    # (5) Apply rotation and translation now
    x_affine = (weight @ x_rescaled.T).T + translation

    # (6) Prediction and cost calculation
    mean, variance = m.predict(x_affine)
    diff = mean - y
    differences_squared = diff ** 2
    mean_diff = np.sum(differences_squared) / n_T
    return mean_diff

def integral_part(alpha, beta_value, x, log_term):
    """ Helper function to compute the integral part of the derivative using quad_vec for better handling of vector inputs. """
    epsilon = 1e-8  # Increased epsilon for improved numerical accuracy, avoiding boundaries

    def integrand(u):
        # Calculate the value of the integrand function
        try:
            value = log_term(u) * u**(alpha - 1) * (1 - u)**(beta_value - 1)
            if not np.isfinite(value):
                return 0  # Handle non-finite value by returning zero
            return value
        except Exception as e:
            return 0

    # Clip x values to avoid instability near boundaries
    x_clipped = np.clip(x, epsilon, 1 - epsilon)

    # Handle scalar or array inputs appropriately
    if np.isscalar(x_clipped):
        # For scalar x, use quad_vec directly
        try:
            result = quad_vec(integrand, epsilon, x_clipped, epsabs=1e-3, epsrel=1e-3)[0]
        except Exception as e:
            warnings.warn(f"Integration failed for x={x_clipped}: {e}")
            result = 0
    else:
        # For vector inputs, iterate over x_clipped to apply quad_vec individually
        result = []
        for xi in x_clipped:
            try:
                res, _ = quad_vec(integrand, epsilon, xi, epsabs=1e-3, epsrel=1e-3)
                result.append(res)
            except Exception as e:
                warnings.warn(f"Integration failed for x={xi}: {e}")
                result.append(0)
        result = np.array(result)
    return result

def beta_cdf_derivative(x, alpha, beta_value):
    """ Compute the derivative of the Beta CDF with respect to alpha and beta. """
    epsilon = 1e-8  # To ensure numerical stability for log and digamma
    
    # Clipping input values
    x_clipped = np.clip(x, 0 + epsilon, 1 - epsilon)
    
    # Computing the integral parts
    dphi_dalpha_integral = integral_part(alpha, beta_value, x_clipped, lambda u: np.log(u))
    dphi_dbeta_integral = integral_part(alpha, beta_value, x_clipped, lambda u: np.log(1 - u))

    # Calculating the Beta CDF at x using scipy.stats.beta
    phi_x = beta_dist.cdf(x_clipped, alpha, beta_value)

    # Derivatives of log(B(alpha, beta))
    dlogB_dalpha = sp.psi(alpha + epsilon) - sp.psi(alpha + beta_value + epsilon)
    dlogB_dbeta = sp.psi(beta_value + epsilon) - sp.psi(alpha + beta_value + epsilon)

    # Compute the Beta function value using scipy.special.betaln
    beta_func_value = math.exp(sp.betaln(alpha, beta_value))

    # Full derivatives including the Beta function part
    dphi_dalpha = dphi_dalpha_integral / beta_func_value - phi_x * dlogB_dalpha
    dphi_dbeta = dphi_dbeta_integral / beta_func_value - phi_x * dlogB_dbeta

    # Handle non-finite results to prevent propagation
    dphi_dalpha = np.nan_to_num(dphi_dalpha, nan=0.0, posinf=50, neginf=-50)
    dphi_dbeta = np.nan_to_num(dphi_dbeta, nan=0.0, posinf=50, neginf=-50)

    return dphi_dalpha, dphi_dbeta

def manifold_gradient_cost_function_with_penalties(
    x, y,
    weight, translation,
    alphas, betas,
    mean, derivative_mean,
    dimension,
    x_beta_transformed_in,    # BetaCDF(x_scaled), in [0,1]
    x_beta_transformed_back,  # BetaCDF(x_scaled) mapped back to [-5,5]
    x_scaled_in               # The [0,1] scaled version of x
    ):
    '''
    The gradient functions for Riemannian transferredGP including rotation, translation, and Beta CDF transformation.
    '''
    n_T = x.shape[0]

    grad_alpha = np.zeros_like(x_scaled_in, dtype=np.float64)
    grad_beta  = np.zeros_like(x_scaled_in, dtype=np.float64)

    # dimension == 1
    if dimension == 1:
        alpha_val = float(alphas[0])
        beta_val  = float(betas[0])

        # x_scaled_in is shape (n_T,1) or (n_T,) => flatten for derivative
        x_scaled_1d = x_scaled_in.flatten()
        dphi_dalpha, dphi_dbeta = beta_cdf_derivative(x_scaled_1d, alpha_val, beta_val)

        # reshape them to (n_T,1)
        dphi_dalpha = dphi_dalpha.reshape(-1, 1)
        dphi_dbeta  = dphi_dbeta.reshape(-1, 1)

        grad_alpha[:, 0] = dphi_dalpha[:, 0]
        grad_beta[:, 0]  = dphi_dbeta[:, 0]

    else:
        # dimension > 1
        for i in range(dimension):
            alpha_val = float(alphas[i])
            beta_val  = float(betas[i])

            dphi_dalpha, dphi_dbeta = beta_cdf_derivative(x_scaled_in[:, i], alpha_val, beta_val)
            dphi_dalpha = dphi_dalpha.reshape(-1)  # ensure shape (n_T,)
            dphi_dbeta  = dphi_dbeta.reshape(-1)

            grad_alpha[:, i] = dphi_dalpha
            grad_beta[:, i]  = dphi_dbeta

    translation_derivative = np.zeros((1, dimension))
    derivative_mean = derivative_mean.reshape((n_T, dimension))
    diff = mean - y

    # Ensure `derivative_mean` has the correct shape
    if derivative_mean.shape[-1] == 1:
        derivative_mean = derivative_mean.squeeze(axis=-1)  # Remove the last dimension
    elif derivative_mean.shape != (n_T, dimension):
        raise ValueError(f"Incompatible shape for derivative_mean: {derivative_mean.shape}")

    # Initialize the gradients for alphas and betas
    loss_grad_alpha = np.zeros(dimension)
    loss_grad_beta = np.zeros(dimension)

    if weight.shape[0] == 1:
        # Treat weight as scalar for element-wise multiplication
        weight = weight.item()  # Convert single-element array to scalar
        result = derivative_mean * weight
    else:
        # Ensure compatible shapes for dot product
        derivative_mean = np.atleast_2d(derivative_mean)
        weight = np.atleast_2d(weight).T
        result = derivative_mean.dot(weight)

    # 2) Build up the elementwise products.
    tmp_alpha = diff[:, None] * result * grad_alpha
    tmp_beta  = diff[:, None] * result * grad_beta

    # 3) Sum over the training samples (axis=0), leaving one value per dimension:
    loss_grad_alpha = (2.0 / n_T) * np.sum(tmp_alpha, axis=0)
    loss_grad_beta  = (2.0 / n_T) * np.sum(tmp_beta, axis=0)

    start_time = time.process_time()  # Start timing for CPU time

    weight_derivative = np.zeros((dimension, dimension))
    for k in range(n_T):
        weight_derivative += diff[k] * np.outer(derivative_mean[k], x_beta_transformed_back[k])
    weight_derivative *= (2 / n_T)
    
    translation_derivative = (2 / n_T) * np.dot(diff.T, derivative_mean)
    translation_derivative = translation_derivative.reshape(1, dimension)

    gradient_calculation_time = time.process_time() - start_time

    projection_weight_derivative = weight @ (0.5 * (weight.T @ weight_derivative - weight_derivative.T @ weight))

    # Extract theta from the rotation matrix for derivative calculation (optional, here set to zero)
    derivative_theta = 0.0
    projection_derivative_theta = 0.0

    returnValue = np.array([projection_weight_derivative, translation_derivative, loss_grad_alpha, loss_grad_beta, derivative_theta, projection_derivative_theta, gradient_calculation_time], dtype=object)

    return returnValue

def iterate_minibatches(inputs, targets, batchsize, j, shuffle=False):
    '''
    The mini batch function.
    '''
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        rng = np.random.default_rng(seed = j)
        rng.shuffle(indices)
    for start_idx in range(0, inputs.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]

def Riemannian_gradient_descent(R, translation, dimension, X_TL_training, Y_TL_training, m, stopping_epoch_threshold, k, 
                                scheduler_type, epochs, batch_size, alpha, decay_rate):
    '''
    The implementation of Riemannian gradient descent method for rotation, translation, and Beta CDF parameters.
    '''
    def initialize(last_rotation=None, current_translation=None, last_alphas=None, last_betas=None, epoch=None, seed=None):
        # Set the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        if last_rotation is None:
            if epoch is not None:
                rotation_seed = k + epoch  # Varies with each epoch
                last_rotation = special_ortho_group.rvs(dim=dimension, random_state=rotation_seed)
            else:
                last_rotation = np.identity(dimension)  # Use identity matrix when epoch is None
        if current_translation is None:
            current_translation = np.random.uniform(-0.5, 0.5, (1, dimension))

            # Ensure `current_translation` is reshaped to (1, dimension)
            if current_translation.ndim == 1:
                current_translation = np.reshape(current_translation, (1, dimension))
        
        if last_alphas is None and last_betas is None:
            # Initialize alphas and betas with more diversity using log-normal with higher variance
            mu_alpha = 0
            sigma_alpha = 0.25  # Higher variance for more diverse initialization
            mu_beta = 1
            sigma_beta = 1

            last_alphas = np.exp(np.random.normal(loc=mu_alpha, scale=sigma_alpha, size=dimension))
            last_betas = np.exp(np.random.normal(loc=mu_beta, scale=sigma_beta, size=dimension))

            print(f"Initialized rotation: {last_rotation}, translation: {current_translation}, alphas: {last_alphas}, betas: {last_betas}")

        # Initialize scheduler
        scheduler = schedulers[scheduler_type.lower()]()

        initial_theta = np.array([last_rotation, current_translation, last_alphas, last_betas], dtype=object)

        return initial_theta, scheduler

    # Define your schedulers
    schedulers = {
        "exponential": lambda: ExponentialScheduler(initial_lr=alpha, decay_rate=decay_rate),
    }

    # Check if scheduler_type is valid
    if scheduler_type.lower() not in schedulers:
        raise ValueError(f"Invalid scheduler_type: {scheduler_type}")

    theta, scheduler = initialize(seed=k)

    previous_cost = None
    adam_history = []
    rotation_matrix_history = []
    translation_matrix_history = []
    alphas_history = []
    betas_history = []
    gradient_rotation_history = []
    projection_gradient_rotation_history = []
    total_cpu_time = 0

    learning_rate_history = []  # Track the learning rate for plotting

    global_best_cost = float('inf')
    global_patience_counter = 0
    patience = 10  # Number of epochs to wait for improvement before stopping
    max_restarts = 6
    restart_count = 0

    for j in range(epochs):
        learning_rate = scheduler(j)
        learning_rate_history.append(learning_rate)  # Track learning rate

        np.random.seed(k + j)
        for batch in iterate_minibatches(X_TL_training, Y_TL_training, batch_size, j, shuffle=True):
            X_mini, y_mini = batch

            # 1) Scale to [0,1]
            problem1_lower_bound = np.full(dimension, -5.) # Define the bound for the source and target problems.
            problem1_upper_bound = np.full(dimension, 5.)
            x_scaled = (X_mini - problem1_lower_bound) / (problem1_upper_bound - problem1_lower_bound)

            # 2) Compute Beta CDF once (dimension-dependent)
            if dimension == 1:
                x_scaled = x_scaled.flatten().astype(np.float64)  # Ensure `x_scaled` is 1D and float64 for dimension = 1
                alphas_value = float(theta[2][0])
                betas_value = float(theta[3][0])
                x_beta_transformed = beta_dist.cdf(x_scaled, alphas_value, betas_value)
            else:
                x_scaled = x_scaled.astype(np.float64)  # Ensure `x_scaled` is of type float64
                x_beta_transformed = np.zeros_like(x_scaled)
                for i in range(dimension):
                    alpha_value = float(theta[2][i])
                    beta_value = float(theta[3][i])
                    x_beta_transformed[:, i] = beta_dist.cdf(x_scaled[:, i], alpha_value, beta_value)
            
            # 3) Map Beta CDF (in [0,1]) back to [-5,5]
            x_beta_transformed_back = x_beta_transformed * (problem1_upper_bound - problem1_lower_bound) + problem1_lower_bound
            if x_beta_transformed_back.ndim == 1:
                x_beta_transformed_back = x_beta_transformed_back.reshape(-1, 1)

            # 4) Apply rotation & translation
            x_transformed = (theta[0] @ x_beta_transformed_back.T).T + theta[1]

            # 5) Forward pass: predict
            mean_training, variance_training = m.predict(x_transformed)
            derivative_mean_training, _ = m.predictive_gradients(x_transformed)

            # 6) Gradient wrt rotation/translation/beta-params
            gradients = manifold_gradient_cost_function_with_penalties(
                x             = X_mini,
                y             = y_mini,
                weight        = theta[0],
                translation   = theta[1],
                alphas        = theta[2],
                betas         = theta[3],
                mean          = mean_training,
                derivative_mean = derivative_mean_training,
                dimension     = dimension,

                # We now pass in what we computed above:
                x_beta_transformed_in  = x_beta_transformed,       # [0,1] range
                x_beta_transformed_back= x_beta_transformed_back,  # [-5,5] range
                x_scaled_in            = x_scaled                  # also [0,1], so we can compute partial derivatives
            )

            grad_weight, grad_translation, grad_alphas, grad_betas, grad_theta, projection_grad_theta, gradient_calc_time = gradients
            gradient_rotation_history.append(grad_theta)
            projection_gradient_rotation_history.append(projection_grad_theta)
            total_cpu_time += gradient_calc_time

            # Update theta using gradients
            skew_symmetric_grad = 0.5 * (grad_weight - grad_weight.T)
            theta[0] = theta[0] @ expm(-learning_rate * skew_symmetric_grad)

            theta[1] = np.reshape(theta[1], (1, dimension))
            grad_translation = np.reshape(grad_translation, (1, dimension))
            theta[1] = theta[1] - (learning_rate * grad_translation)

            # Reshape grad_alphas and grad_betas to ensure they are 2D
            grad_alphas = np.array(grad_alphas).reshape(-1, dimension)  # Shape (n_samples, dimension)
            grad_betas = np.array(grad_betas).reshape(-1, dimension)    # Shape (n_samples, dimension)

            # Calculate averages
            for i in range(dimension):
                grad_alpha_avg_i = np.mean(grad_alphas[:, i])  # Average over samples
                grad_beta_avg_i = np.mean(grad_betas[:, i])    # Average over samples
                theta[2][i] -= learning_rate * grad_alpha_avg_i
                theta[3][i] -= learning_rate * grad_beta_avg_i

            # Apply constraints
            theta[2] = np.maximum(theta[2], 1e-8)  # Ensure all elements in theta[2] are at least 1e-10
            theta[3] = np.maximum(theta[3], 1e-8)  # Ensure all elements in theta[3] are at least 1e-10

        # Track history
        current_cost = cost_function(X_TL_training, Y_TL_training, theta[2], theta[3], theta[0], theta[1], m, dimension)
        adam_history.append(current_cost)
        rotation_matrix_history.append(theta[0])
        translation_matrix_history.append(theta[1])
        alphas_history.append(theta[2].copy())
        betas_history.append(theta[3].copy())

        # # Check for restart
        # if previous_cost and np.isclose(previous_cost, current_cost):
        #     print(f"Restarting at epoch {j} due to small cost change: {previous_cost} to {current_cost}")
        #     theta, scheduler = initialize(epoch=j)
        #     previous_cost = None
        #     continue

        # Check global early stopping
        if current_cost < global_best_cost:
            global_best_cost = current_cost
            global_patience_counter = 0  # Reset patience
        else:
            global_patience_counter += 1

        if global_patience_counter >= patience:
            if restart_count < max_restarts:
                print(f"Random restart #{restart_count + 1}")
                theta, scheduler = initialize(epoch=j)
                global_patience_counter = 0  # Reset patience
                restart_count += 1
            else:
                print("Early stopping triggered after max restarts!")
                break

        previous_cost = current_cost

    # Final output
    # for idx, (alpha_values, beta_values) in enumerate(zip(alphas_history, betas_history)):
    #     print(f"Epoch {idx}: alphas={alpha_values}, betas={beta_values}")

    print("The number of iterations for training: ", len(adam_history))
    index_minimum_cost = np.argmin(adam_history)
    final_cost = adam_history[index_minimum_cost]
    final_rotation_matrix = rotation_matrix_history[index_minimum_cost]
    final_translation_matrix = translation_matrix_history[index_minimum_cost]
    final_alphas = alphas_history[index_minimum_cost]
    final_betas = betas_history[index_minimum_cost]

    print("The minimum loss value: ", final_cost)
    print("The determinant of the generated rotation matrix: ", np.linalg.det(final_rotation_matrix))
    print('\nFinal rotation matrix = {}'.format(final_rotation_matrix),
          '\nFinal translation matrix = {}'.format(final_translation_matrix),
          '\nFinal alpha = {}'.format(final_alphas),
          '\nFinal beta = {}'.format(final_betas))

    # print(alphas_history)
    # print(betas_history)

    return final_cost, final_rotation_matrix, final_translation_matrix, final_alphas, final_betas, \
           adam_history, rotation_matrix_history, translation_matrix_history, alphas_history, betas_history, \
           total_cpu_time, learning_rate_history
    

class AutoML():
    '''
    The implementations of the hyper parameter tuning of the Riemannian gradient descent method, including Beta CDF parameters.
    '''
    def __init__(self, R, translation, alphas, betas, dimension, X_TL_training, Y_TL_training, X_TL_test, Y_TL_test, m, stopping_epoch_threshold, k, scheduler_type, epochs, batch_size, 
                 alpha, decay_rate):
        self.R = R
        self.translation = translation
        self.alphas = alphas
        self.betas = betas
        self.dimension = dimension
        self.alpha = alpha
        self.epochs = epochs
        self.X_TL_training = X_TL_training
        self.Y_TL_training = Y_TL_training
        self.batch_size = batch_size
        self.m = m
        self.stopping_epoch_threshold = stopping_epoch_threshold
        self.decay_rate = decay_rate
        self.scheduler_type = scheduler_type
        self.k = k
        self.X_TL_test = X_TL_test
        self.Y_TL_test = Y_TL_test

    def configspace(self, seed) -> ConfigurationSpace:
        # Build the configuration space with all parameters.
        cs = ConfigurationSpace(seed=seed)

        # Create the hyperparameters with their range
        lr = Float("lr", (1e-7, 1), default=0.001, log=True)
        decay_rate = Float("decay_rate", (5e-3, 1), default=0.1)
        epochs = Integer("epochs", (50, 100), default=60)
        if self.batch_size == 1:
            batch_size = Integer("batch_size", (1, 2), default=1)
        else:
            batch_size = Integer("batch_size", (math.ceil(self.batch_size / 2), self.batch_size), default=self.batch_size)

        cs.add_hyperparameters([lr, decay_rate, epochs, batch_size])

        # Print the hyperparameters for debugging or further checks
        print("Hyperparameter Configuration:")
        for hyperparameter in cs.get_hyperparameters():
            print(f"  - {hyperparameter.name}: Range=({hyperparameter.lower}, {hyperparameter.upper}), Default={hyperparameter.default_value}")

        return cs
    
    def train(self, config: Configuration, seed: int) -> float:
        config_dict = config.get_dictionary()

        # Use Riemannian gradient descent to train with rotation, translation, and Beta CDF parameters
        final_cost, final_rotation_matrix, final_translation_matrix, final_alphas, final_betas, adam_history, rotation_history, translation_history, alphas_history, betas_history, total_cpu_time, learning_rate_history = Riemannian_gradient_descent(
            R=self.R, 
            translation=self.translation, 
            dimension=self.dimension, 
            X_TL_training=self.X_TL_training, 
            Y_TL_training=self.Y_TL_training, 
            m=self.m, 
            stopping_epoch_threshold=self.stopping_epoch_threshold, 
            k=self.k, 
            scheduler_type=self.scheduler_type, 
            epochs=config_dict['epochs'], 
            batch_size=config_dict['batch_size'], 
            alpha=config_dict['lr'], 
            decay_rate=config_dict['decay_rate']
        )

        # Evaluate the model on the test set
        problem1_lower_bound = np.full(self.dimension, -5.) # Define the bound for the source and target problems.
        problem1_upper_bound = np.full(self.dimension, 5.)
        x_scaled_test = (self.X_TL_test - problem1_lower_bound) / (problem1_upper_bound - problem1_lower_bound)

        # Ensure `x_scaled_test` is float64 for compatibility with Beta CDF
        x_scaled_test = x_scaled_test.astype(np.float64)

        # Apply Beta CDF transformation
        if self.dimension == 1:
            # If dimension is 1, ensure x_scaled_test is treated correctly
            x_scaled_test = x_scaled_test.flatten()
            x_beta_transformed_test = beta_dist.cdf(x_scaled_test, float(final_alphas[0]), float(final_betas[0]))
        else:
            x_beta_transformed_test = np.zeros_like(x_scaled_test)
            for i in range(self.dimension):
                x_beta_transformed_test[:, i] = beta_dist.cdf(x_scaled_test[:, i], final_alphas[i], final_betas[i])

        # Transform back to original range [-5, 5]
        if self.dimension == 1:
            x_transformed_test = x_beta_transformed_test * (problem1_upper_bound[0] - problem1_lower_bound[0]) + problem1_lower_bound[0]
            x_transformed_test = x_transformed_test.reshape(-1, 1)  # Ensure it is a 2D column vector
        else:
            x_transformed_test = x_beta_transformed_test * (problem1_upper_bound - problem1_lower_bound) + problem1_lower_bound

        x_transformed_test = (final_rotation_matrix @ x_transformed_test.T).T + final_translation_matrix
        mean_TL, variance_TL = self.m.predict(x_transformed_test)
        symmetric_mean_absolute_percentage_error = smape(self.Y_TL_test, mean_TL)

        return symmetric_mean_absolute_percentage_error

def AutoML_Riemannian_gradient_descent(R, translation, alphas, betas, dimension, X_TL_training, Y_TL_training, X_TL_test, Y_TL_test,
                                       m, stopping_epoch_threshold, scheduler_type, epochs, batch_size, alpha, decay_rate, k, save_path):
    # Initialize the AutoML class with parameters including alphas and betas
    Riemannian_SGD = AutoML(R, translation, alphas, betas, dimension, X_TL_training, Y_TL_training, X_TL_test, Y_TL_test, m, stopping_epoch_threshold, k, scheduler_type, epochs, batch_size, alpha, decay_rate)
    
    # Create a configurationSpace instance
    cs = Riemannian_SGD.configspace(k)
    
    # Specify the optimization environment with a defined n_trials
    scenario = Scenario(cs, deterministic=True, n_trials=250, output_directory=save_path, seed=k)

    # Use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, Riemannian_SGD.train)
    incumbent = smac.optimize()
    
    print("Incumbent lr :", incumbent["lr"])
    print("Incumbent decay_rate :", incumbent["decay_rate"])
    print("Incumbent batch_size :", incumbent["batch_size"])
    print("Incumbent epochs :", incumbent["epochs"])
    
    # Use the best found hyperparameters to train the final model
    final_cost, final_rotation_matrix, final_translation_matrix, final_alphas, final_betas, adam_history, rotation_history, translation_history, alphas_history, betas_history, total_cpu_time, learning_rate_history = Riemannian_gradient_descent(
        R=R, 
        translation=translation, 
        dimension=dimension, 
        X_TL_training=X_TL_training, 
        Y_TL_training=Y_TL_training, 
        m=m, 
        stopping_epoch_threshold=stopping_epoch_threshold, 
        k=k, 
        scheduler_type=scheduler_type, 
        epochs=incumbent["epochs"], 
        batch_size=incumbent["batch_size"], 
        alpha=incumbent["lr"], 
        decay_rate=incumbent["decay_rate"]
    )
    
    return final_cost, final_rotation_matrix, final_translation_matrix, final_alphas, final_betas, adam_history, rotation_history, translation_history, alphas_history, betas_history, total_cpu_time, learning_rate_history