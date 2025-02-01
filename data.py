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


def generate_data_affine_betacdf_transformation(problem_selection, dimension, number_of_data_GPR_training, number_of_data_GPR_test,
                  number_of_data_TL_training, number_of_data_TL_test, k, save_path):
    '''
    Generate the data from the source and target problems using affine transformation and betacdf.
    '''
    problem1 = ioh.get_problem(
        problem_selection,
        instance=1,
        dimension=dimension,
        problem_type=ioh.ProblemType.REAL
    )
    
    # Define the bound for the source and target problems
    problem1_lower_bound = np.full(dimension, -5.)
    problem1_upper_bound = np.full(dimension, 5.)

    # Generate full data set for original GPR
    X_GPR_full = np.random.uniform(problem1_lower_bound, problem1_upper_bound, (number_of_data_GPR_training +
                                                                            number_of_data_GPR_test, dimension))
    Y_GPR_full = np.array([problem1(x) - problem1.optimum.y for x in X_GPR_full])
    Y_GPR_full_transformed = np.log10(Y_GPR_full + 1).reshape(-1, 1)
    print("Size of X_GPR_full:", X_GPR_full.shape)
    print("Size of Y_GPR_full_transformed:", Y_GPR_full_transformed.shape)
    X_GPR_training, X_GPR_test, Y_GPR_training, Y_GPR_test = train_test_split(X_GPR_full, Y_GPR_full_transformed, 
                                                                    test_size=number_of_data_GPR_test, random_state=k)
    del X_GPR_full, Y_GPR_full, Y_GPR_full_transformed

    # (1) Generate data set for transfer learning process
    X_TL_full = np.random.uniform(problem1_lower_bound, problem1_upper_bound, (number_of_data_TL_training + number_of_data_TL_test, dimension))
                    
    # (2) Scale to [0,1]
    X_scaled = (X_TL_full - problem1_lower_bound) / (problem1_upper_bound - problem1_lower_bound)

    # (3) Apply the Beta CDF transformation
    # Generate alphas and betas using log-normal distribution
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
    X_TL_transformed_rescaled = X_TL_transformed * (problem1_upper_bound - problem1_lower_bound) + problem1_lower_bound

    # (5) Apply rotation and translation as a final step
    R = special_ortho_group.rvs(dim=dimension, random_state=k)
    print("The generated rotation matrix: ", R)
    translation = np.random.uniform(-0.5, 0.5, (1, dimension))
    print("The generated translation matrix: ", translation)
    X_TL_final = (R.T @ (X_TL_transformed_rescaled - translation).T).T

    Y_TL_affined_full = np.array([problem1(x) - problem1.optimum.y for x in X_TL_final])
    Y_TL_full_transformed = np.log10(Y_TL_affined_full + 1).reshape(-1, 1)
    
    X_TL_training, X_TL_test, Y_TL_training, Y_TL_test = train_test_split(X_TL_full, Y_TL_full_transformed, 
                                                            test_size=number_of_data_TL_test, random_state=k)

    del X_TL_full, X_scaled, X_TL_transformed, X_TL_transformed_rescaled, Y_TL_affined_full, Y_TL_full_transformed

    TL_training_data = np.concatenate((X_TL_training, Y_TL_training), axis=1)

    return X_GPR_training, Y_GPR_training, X_GPR_test, Y_GPR_test, X_TL_training, Y_TL_training, \
        X_TL_test, Y_TL_test, TL_training_data, R, translation, alphas, betas, problem1