#!/usr/bin/env python
# coding: utf-8
'''
Script Name: data.py
Description: This script mainly works for generate the data for training and evaluating the origianl GPR,
the transfer data, and the test date generated from the target function.
'''
import os
import ioh
import numpy as np
from scipy.stats import special_ortho_group
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from decimal import *
from utils import *
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import beta
from shapely.geometry import Polygon, Point


def check_if_in_original_domain(points, R, translation, bounds_lower, bounds_upper):
    transformed_back_points = (R.T @ points.T).T + translation
    return np.all((transformed_back_points >= bounds_lower) & (transformed_back_points <= bounds_upper), axis=1)

def count_in_domain(points, lower_bound, upper_bound):
    ''' Checks if points are within the given bounds '''
    return np.all((points >= lower_bound) & (points <= upper_bound), axis=1)

def generate_data_in_domain(problem_selection, dimension, number_of_data_GPR_training, number_of_data_GPR_test,
                              number_of_data_TL_training, number_of_data_TL_test, k, save_path):
    problem = ioh.get_problem(problem_selection, instance=1, dimension=dimension, problem_type=ioh.ProblemType.REAL)
    bounds_lower = np.full(dimension, -5.)
    bounds_upper = np.full(dimension, 5.)

    X_GPR = np.random.uniform(bounds_lower, bounds_upper, (number_of_data_GPR_training + number_of_data_GPR_test, dimension))
    Y_GPR = np.array([problem(x) - problem.optimum.y for x in X_GPR])
    Y_GPR_transformed = np.log10(Y_GPR + 1).reshape(-1, 1)
    X_GPR_training, X_GPR_test, Y_GPR_training, Y_GPR_test = train_test_split(X_GPR, Y_GPR_transformed, test_size=number_of_data_GPR_test, random_state=k)

    # Generate TL data and apply affine transformation
    X_TL = np.random.uniform(bounds_lower, bounds_upper, (number_of_data_TL_training + number_of_data_TL_test, dimension))

    # (2) Scale to [0,1]
    X_scaled = (X_TL - bounds_lower) / (bounds_upper - bounds_lower)

    # (3) Apply the Beta CDF transformation
    mu_alpha = 0  # Mean for the log of alpha
    sigma_alpha = 0.25  # Standard deviation for the log of alpha
    mu_beta = 1  # Mean for the log of beta
    sigma_beta = 1  # Standard deviation for the log of beta

    alphas = np.exp(np.random.normal(loc=mu_alpha, scale=sigma_alpha, size=dimension))
    betas = np.exp(np.random.normal(loc=mu_beta, scale=sigma_beta, size=dimension))
    print("The generated alphas of betacdf (log-normal): ", alphas)
    print("The generated betas of betacdf (log-normal): ", betas)

    X_TL_transformed = np.zeros_like(X_scaled)
    for i in range(dimension):
        X_TL_transformed[:, i] = beta.cdf(X_scaled[:, i], alphas[i], betas[i])

    # (4) Rescale back to [-5, 5]
    X_TL_transformed_rescaled = X_TL_transformed * (bounds_upper - bounds_lower) + bounds_lower
    
    R = special_ortho_group.rvs(dim=dimension, random_state=k)
    print("The generated rotation matrix: ", R)
    translation = np.random.uniform(-0.5, 0.5, dimension)
    print("The generated translation matrix: ", translation)
    X_TL_transformed = (R @ (X_TL_transformed_rescaled - translation).T).T

    Y_TL_affined_full = np.array([problem(x) - problem.optimum.y for x in X_TL_transformed]) # Vectorized problem evaluation
    Y_TL_full_transformed = np.log10(Y_TL_affined_full + 1).reshape(-1, 1)
    X_TL_training, X_TL_test, Y_TL_training, Y_TL_test = train_test_split(X_TL, Y_TL_full_transformed, test_size=number_of_data_TL_test, random_state=k)

    inverse_in_domain = check_if_in_original_domain(X_GPR, R, translation, bounds_lower, bounds_upper)
    inside_original_domain_count = np.sum(inverse_in_domain)
    percentage_inside_original_domain = (inside_original_domain_count / len(X_GPR)) * 100
    print("Number of source points are in-domain for the target function:", inside_original_domain_count)
    print(f"Percentage of source points are in-domain for the target function: {percentage_inside_original_domain:.2f}%")

    in_domain_after_transformation = count_in_domain(X_TL_transformed, bounds_lower, bounds_upper)
    in_domain_after_transformation_count = np.sum(in_domain_after_transformation)
    percent_in_domain_after_transformation = 100 * np.mean(in_domain_after_transformation)
    print("Number of transfer points that are in-domain for the source:", in_domain_after_transformation_count)
    print(f"Percentage of transfer points that are still in-domain for the source: {percent_in_domain_after_transformation:.2f}%")

    X_TL_training_transformed = (R @ (X_TL_training - translation).T).T
    in_domain_training_mask = count_in_domain(X_TL_training_transformed, bounds_lower, bounds_upper)
    X_TL_training_in_domain = X_TL_training[in_domain_training_mask]
    Y_TL_training_in_domain = Y_TL_training[in_domain_training_mask]

    # Logging the filtered counts
    print("Original training points:", len(X_TL_training))
    print("In-domain training points:", len(X_TL_training_in_domain))
    print("Evaluation points (unchanged):", len(X_TL_test))

    del X_GPR, Y_GPR, Y_GPR_transformed
    del X_TL, Y_TL_affined_full, Y_TL_full_transformed # Clear memory

    # Return in-domain training points and original test points
    TL_training_data = np.concatenate((X_TL_training_in_domain, Y_TL_training_in_domain), axis=1)

    return X_GPR_training, Y_GPR_training, X_GPR_test, Y_GPR_test, X_TL_training_in_domain, Y_TL_training_in_domain, \
        X_TL_test, Y_TL_test, TL_training_data, R, translation, alphas, betas, problem