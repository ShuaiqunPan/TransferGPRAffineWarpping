#!/usr/bin/env python
# coding: utf-8
import os
num_threads = "1"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import numpy as np
import matplotlib.pyplot as plt
from surrogate import *
from data import *
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Arc
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def scientific_notation(x, pos):
    return "{:.1e}".format(x)

def main(problem_selection):
    dimension = 2

    number_of_data_GPR_training = 1000 * dimension
    number_of_data_GPR_test = 1000 * dimension

    number_of_data_TL_training = 40
    max_number_of_data_TL_training = 40 * dimension
    number_of_data_TL_test = 1000 * dimension

    load_path = 'Result/'
    save_path = load_path + str(dimension) + '/' + f"{problem_selection}" + '/' + str(number_of_data_TL_training)

    number_of_repetition = 10
    for k in range(1, number_of_repetition+1):
        np.random.seed(k)
        print('-------------------------------------------------------------------------------------------------')
        print("Number of repetition: ", k)
        print('-------------------------------------------------------------------------------------------------')

        X_GPR_training, Y_GPR_training, X_GPR_test, Y_GPR_test, X_TL_training, Y_TL_training, \
            X_TL_test, Y_TL_test, TL_training_data, R, translation, alphas, betas, problem1 = generate_data_affine_betacdf_transformation(problem_selection,
                                                                                            dimension,
                                                                                            number_of_data_GPR_training,
                                                                                            number_of_data_GPR_test,
                                                                                            max_number_of_data_TL_training,
                                                                                            number_of_data_TL_test, k, save_path)

        train_data_generation = TL_training_data[np.random.choice(TL_training_data.shape[0], number_of_data_TL_training, replace=False),:]
        X_TL_training1 = train_data_generation[:, :dimension]
        Y_TL_training1 = train_data_generation[:, [-1]]
        print(X_TL_training1.shape)
    
        # Plot individual contour line for each repetition
        save_original_model_GP = load_path + str(dimension) + '/' + f"{problem_selection}" + '/original_model'
        complete_name_GP = os.path.join(save_original_model_GP, f"{problem_selection}_{k}"'.npy')
        m_GP = load_GP_model(X_GPR_training, Y_GPR_training, complete_name_GP)
        mean_original_GP, variance_original_GP = m_GP.predict(X_TL_test)
        MAPE_original_GP, SMAPE_original_GP, R_square_original_GP, log_ratio_original_GP, square_sum_log_ratio_original_GP = prediction(mean_original_GP, Y_TL_test)

        save_trained_model_GP_40 = load_path + str(dimension) + '/' + f"{problem_selection}" + '/40/without_transfer'
        complete_name_GP = os.path.join(save_trained_model_GP_40, f"{problem_selection}_{k}"'.npy')
        m_GP_trained_40 = load_GP_model(X_TL_training1, Y_TL_training1, complete_name_GP)
        mean_trained_model_GP_40, variance_trained_model_GP_40 = m_GP_trained_40.predict(X_TL_test)
        MAPE_trained_model_GP_40, SMAPE_trained_model_GP_40, R_square_trained_model_GP_40, log_ratio_trained_model_GP_40, square_sum_log_ratio_trained_model_GP_40 = prediction(mean_trained_model_GP_40, Y_TL_test)
        
        save_trained_model_GP_80 = load_path + str(dimension) + '/' + f"{problem_selection}" + '/80/without_transfer'
        complete_name_GP = os.path.join(save_trained_model_GP_80, f"{problem_selection}_{k}"'.npy')
        m_GP_trained_80 = load_GP_model(X_TL_training, Y_TL_training, complete_name_GP)
        mean_trained_model_GP_80, variance_trained_model_GP_80 = m_GP_trained_80.predict(X_TL_test)
        MAPE_trained_model_GP_80, SMAPE_trained_model_GP_80, R_square_trained_model_GP_80, log_ratio_trained_model_GP_80, square_sum_log_ratio_trained_model_GP_80 = prediction(mean_trained_model_GP_80, Y_TL_test)

        # Transfer model with 40 data points
        save_AutoML_RSGD_40 = load_path + str(dimension) + '/' + f"{problem_selection}" + '/40/transfer_model'
        final_rotation_matrix_complete_name_RSGD_40 = os.path.join(save_AutoML_RSGD_40, f"final_rotation_{problem_selection}_{k}"'.npy')
        final_translation_matrix_complete_name_RSGD_40 = os.path.join(save_AutoML_RSGD_40, f"final_translation_{problem_selection}_{k}"'.npy')
        final_alphas_complete_name_RSGD_40 = os.path.join(save_AutoML_RSGD_40, f"final_alphas_{problem_selection}_{k}.npy")
        final_betas_complete_name_RSGD_40 = os.path.join(save_AutoML_RSGD_40, f"final_betas_{problem_selection}_{k}.npy")
        final_rotation_matrix_RSGD_40 = np.load(final_rotation_matrix_complete_name_RSGD_40)
        final_translation_matrix_RSGD_40 = np.load(final_translation_matrix_complete_name_RSGD_40)
        final_alpha_RSGD_40 = np.load(final_alphas_complete_name_RSGD_40)
        final_betas_RSGD_40 = np.load(final_betas_complete_name_RSGD_40)

        problem1_lower_bound = np.full(dimension, -5.)
        problem1_upper_bound = np.full(dimension, 5.)
        x_scaled_test_GP_40 = (X_TL_test - problem1_lower_bound) / (problem1_upper_bound - problem1_lower_bound)
        # Apply Beta CDF transformation for each dimension
        x_beta_transformed_test_GP_40 = np.zeros_like(x_scaled_test_GP_40)
        for i in range(x_scaled_test_GP_40.shape[1]):
            x_beta_transformed_test_GP_40[:, i] = beta.cdf(x_scaled_test_GP_40[:, i], final_alpha_RSGD_40[i], final_betas_RSGD_40[i])
        x_transformed_test_GP_40 = x_beta_transformed_test_GP_40 * (problem1_upper_bound - problem1_lower_bound) + problem1_lower_bound

        # Apply the affine transformation: rotation and translation
        x_transformed_test_GP_40 = (final_rotation_matrix_RSGD_40 @ x_transformed_test_GP_40.T).T + final_translation_matrix_RSGD_40

        mean_transfer_GP_40, variance_transfer_GP_40 = m_GP.predict(x_transformed_test_GP_40)
        MAPE_transfer_GP_40, SMAPE_transfer_GP_40, R_square_transfer_GP_40, log_ratio_transfer_GP_40, square_sum_log_ratio_transfer_GP_40 = prediction(mean_transfer_GP_40, Y_TL_test)

        # Transfer model with 80 data points
        save_AutoML_RSGD_80 = load_path + str(dimension) + '/' + f"{problem_selection}" + '/80/transfer_model'
        final_rotation_matrix_complete_name_RSGD_80 = os.path.join(save_AutoML_RSGD_80, f"final_rotation_{problem_selection}_{k}"'.npy')
        final_translation_matrix_complete_name_RSGD_80 = os.path.join(save_AutoML_RSGD_80, f"final_translation_{problem_selection}_{k}"'.npy')
        final_alphas_complete_name_RSGD_80 = os.path.join(save_AutoML_RSGD_80, f"final_alphas_{problem_selection}_{k}.npy")
        final_betas_complete_name_RSGD_80 = os.path.join(save_AutoML_RSGD_80, f"final_betas_{problem_selection}_{k}.npy")
        final_rotation_matrix_RSGD_80 = np.load(final_rotation_matrix_complete_name_RSGD_80)
        final_translation_matrix_RSGD_80 = np.load(final_translation_matrix_complete_name_RSGD_80)
        final_alpha_RSGD_80 = np.load(final_alphas_complete_name_RSGD_80)
        final_betas_RSGD_80 = np.load(final_betas_complete_name_RSGD_80)
        
        problem1_lower_bound = np.full(dimension, -5.)
        problem1_upper_bound = np.full(dimension, 5.)
        x_scaled_test_GP_80 = (X_TL_test - problem1_lower_bound) / (problem1_upper_bound - problem1_lower_bound)
        # Apply Beta CDF transformation for each dimension
        x_beta_transformed_test_GP_80 = np.zeros_like(x_scaled_test_GP_80)
        for i in range(x_scaled_test_GP_80.shape[1]):
            x_beta_transformed_test_GP_80[:, i] = beta.cdf(x_scaled_test_GP_80[:, i], final_alpha_RSGD_80[i], final_betas_RSGD_80[i])
        x_transformed_test_GP_80 = x_beta_transformed_test_GP_80 * (problem1_upper_bound - problem1_lower_bound) + problem1_lower_bound
        x_transformed_test_GP_80 = (final_rotation_matrix_RSGD_80 @ x_transformed_test_GP_80.T).T + final_translation_matrix_RSGD_80
        mean_transfer_GP_80, variance_transfer_GP_80 = m_GP.predict(x_transformed_test_GP_80)
        MAPE_transfer_GP_80, SMAPE_transfer_GP_80, R_square_transfer_GP_80, log_ratio_transfer_GP_80, square_sum_log_ratio_transfer_GP_80 = prediction(mean_transfer_GP_80, Y_TL_test)

        X, Y = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
        Z = np.zeros(X.shape)
        Z_affined = np.zeros(X.shape)
        original_GPR_Z_affined = np.zeros(X.shape)
        Trained_GPR_Z_affined_40 = np.zeros(X.shape)
        Trained_GPR_Z_affined_80 = np.zeros(X.shape)
        TL_GPR_Z_affined_40 = np.zeros(X.shape)
        TL_GPR_Z_affined_80 = np.zeros(X.shape)

        for idx1 in range(100):
            for idx2 in range(100):
                data = [X[idx1, idx2], Y[idx1, idx2]]
                data_array = np.array(data).reshape(1, dimension)

                problem1_lower_bound = np.full(dimension, -5.)
                problem1_upper_bound = np.full(dimension, 5.)
                data_affine = (data_array - problem1_lower_bound) / (problem1_upper_bound - problem1_lower_bound)
                data_beta_transformed = np.zeros_like(data_affine)
                for i in range(dimension):
                    data_beta_transformed[:, i] = beta.cdf(data_affine[:, i], alphas[i], betas[i])
                data_affine = data_beta_transformed * (problem1_upper_bound - problem1_lower_bound) + problem1_lower_bound
                data_affine = (R.T @ (data_affine - translation).T).T

                Z[idx1, idx2] = problem1(data)
                Z_affined[idx1, idx2] = problem1(data_affine[0])
                
                original_GPR_Z_affined[idx1, idx2] = 10.0 ** (m_GP.predict(data_array)[0].item())

                Trained_GPR_Z_affined_40[idx1, idx2] = 10.0 ** (m_GP_trained_40.predict(data_array)[0].item())
                Trained_GPR_Z_affined_80[idx1, idx2] = 10.0 ** (m_GP_trained_80.predict(data_array)[0].item())

                data_array_affine = (data_array - problem1_lower_bound) / (problem1_upper_bound - problem1_lower_bound)
                data_array_beta_transformed = np.zeros_like(data_array_affine)
                for i in range(data_array_affine.shape[1]):
                    data_array_beta_transformed[:, i] = beta.cdf(data_array_affine[:, i], final_alpha_RSGD_40[i], final_betas_RSGD_40[i])
                data_array_beta_transformed = data_array_beta_transformed * (problem1_upper_bound - problem1_lower_bound) + problem1_lower_bound
                data_array_beta_transformed = (final_rotation_matrix_RSGD_40 @ data_array_beta_transformed.T).T + final_translation_matrix_RSGD_40
                TL_GPR_Z_affined_40[idx1, idx2] = 10.0 ** m_GP.predict(data_array_beta_transformed)[0].item()
                
                data_array_affine = (data_array - problem1_lower_bound) / (problem1_upper_bound - problem1_lower_bound)
                data_array_beta_transformed = np.zeros_like(data_array_affine)
                for i in range(data_array_affine.shape[1]):
                    data_array_beta_transformed[:, i] = beta.cdf(data_array_affine[:, i], final_alpha_RSGD_80[i], final_betas_RSGD_80[i])
                data_array_beta_transformed = data_array_beta_transformed * (problem1_upper_bound - problem1_lower_bound) + problem1_lower_bound
                data_array_beta_transformed = (final_rotation_matrix_RSGD_80 @ data_array_beta_transformed.T).T + final_translation_matrix_RSGD_80
                TL_GPR_Z_affined_80[idx1, idx2] = 10.0 ** m_GP.predict(data_array_beta_transformed)[0].item()

        Z_list = [Z, Z_affined, original_GPR_Z_affined, Trained_GPR_Z_affined_40, TL_GPR_Z_affined_40, Trained_GPR_Z_affined_80, TL_GPR_Z_affined_80]

        # Step 1: Calculate global min and max
        global_min = min([np.min(z) for z in Z_list])
        global_max = max([np.max(z) for z in Z_list])

        # Step 2: Create subplots
        figure2, axes = plt.subplots(1, 7, figsize=(42, 6), sharey=True, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})

        axes_list = []
        for j in range(7):
            axes_list.append(axes[j])

        for i, ax in enumerate(axes_list):
            im = ax.contourf(X, Y, Z_list[i], levels=80)

            # Set font size for axis labels (x-axis and y-axis)
            ax.tick_params(axis='both', which='major', labelsize=24)  # Adjust '14' to your desired font size
            
            func_names = {
                1: "Sphere",
                2: "Ellipsoid",
                3: "Rastrigin",
                4: "BuecheRastrigin",
                5: "LinearSlope",
                6: "AttractiveSector",
                7: "StepEllipsoid",
                8: "Rosenbrock",
                9: "RosenbrockRotated",
                10: "EllipsoidRotated",
                11: "Discus",
                12: "BentCigar",
                13: "SharpRidge",
                14: "DifferentPowers",
                15: "RastriginRotated",
                16: "Weierstrass",
                17: "Schaffers10",
                18: "Schaffers1000",
                19: "GriewankRosenBrock",
                20: "Schwefel",
                21: "Gallagher101",
                22: "Gallagher21",
                23: "Katsuura",
                24: "LunacekBiRastrigin",
            } # Dictionary mapping problem_selection values to function names
            
            reverse_func_names = {v: k for k, v in func_names.items()} # Reverse the dictionary
            func_name = reverse_func_names.get(problem_selection, 1) # Get function name from dictionary

            if i == 0:
                # ax.text(-4.5, 3.5, "Source function", color='white', fontsize=36)
                ax.text(-4.5, 3, f"F{func_name}", color='white', fontsize=36)
            # elif i == 1:
            #     ax.text(-4.5, 3.5, "Target function", color='white', fontsize=36)
            #     ax.text(-4.5, 1.6, f"Vertical (x1):\nTarget Alpha = {alphas[1]:.2f}, Beta = {betas[1]:.2f}", color='white', fontsize=36)
            #     ax.text(-4.5, 0.1, f"Horizontal (x2):\nTarget Alpha = {alphas[0]:.2f}, Beta = {betas[0]:.2f}", color='white', fontsize=36)
            elif i == 2:
                # ax.text(-4.5, 3.5, "Original GPR", color='white', fontsize=36)
                ax.text(-4.5, 3, f"SMAPE: {round_number(SMAPE_original_GP)}", color='white', fontsize=36)
            elif i == 3:
                # ax.text(-4.5, 3.5, "Trained with 40 samples", color='white', fontsize=36)
                ax.text(-4.5, 3, f"SMAPE: {round_number(SMAPE_trained_model_GP_40)}", color='white', fontsize=36)
            elif i == 4:
                # ax.text(-4.5, 3.5, "Transferred with 40 samples", color='white', fontsize=36)
                ax.text(-4.5, 3, f"SMAPE: {round_number(SMAPE_transfer_GP_40)}", color='white', fontsize=36)
                # ax.text(-4.5, 1.6, f"Vertical (x1):\nLearned Alpha = {final_alpha_RSGD_40[1]:.2f}, Beta = {final_betas_RSGD_40[1]:.2f}", color='white', fontsize=36)
                # ax.text(-4.5, 0.1, f"Horizontal (x2):\nLearned Alpha = {final_alpha_RSGD_40[0]:.2f}, Beta = {final_betas_RSGD_40[0]:.2f}", color='white', fontsize=36)
            elif i == 5:
                # ax.text(-4.5, 3.5, "Trained with 80 samples", color='white', fontsize=36)
                ax.text(-4.5, 3, f"SMAPE: {round_number(SMAPE_trained_model_GP_80)}", color='white', fontsize=36)
            elif i == 6:
                # ax.text(-4.5, 3.5, "Transferred with 80 samples", color='white', fontsize=36)
                ax.text(-4.5, 3, f"SMAPE: {round_number(SMAPE_transfer_GP_80)}", color='white', fontsize=36)
                # ax.text(-4.5, 1.6, f"Vertical (x1):\nLearned Alpha = {final_alpha_RSGD_80[1]:.2f}, Beta = {final_betas_RSGD_80[1]:.2f}", color='white', fontsize=36)
                # ax.text(-4.5, 0.1, f"Horizontal (x2):\nLearned Alpha = {final_alpha_RSGD_80[0]:.2f}, Beta = {final_betas_RSGD_80[0]:.2f}", color='white', fontsize=36)

                formatter = FuncFormatter(scientific_notation)
    
            # Step 5: Add colorbar to the last plot
            if i == 6:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = figure2.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.yaxis.set_ticks([])

        plt.tight_layout()

        complete_name = os.path.join(save_path, f"{problem_selection}_countour_line_latest_{k}.pdf")
        plt.savefig(complete_name)
        plt.close()
        
if __name__ == '__main__':
    BBOB_function_list = ['Sphere', 'Ellipsoid', 'Rastrigin', 'BuecheRastrigin', 'LinearSlope', 'AttractiveSector', 'StepEllipsoid', 
                          'Rosenbrock', 'RosenbrockRotated', 'EllipsoidRotated', 'Discus', 'BentCigar', 'SharpRidge', 'DifferentPowers', 
                          'RastriginRotated', 'Weierstrass', 'Schaffers10', 'Schaffers1000', 'GriewankRosenBrock', 'Schwefel', 'Gallagher101', 
                          'Gallagher21', 'Katsuura', 'LunacekBiRastrigin']
    pool = Pool()
    result = pool.map(main, BBOB_function_list)
    pool.close()
    pool.join()