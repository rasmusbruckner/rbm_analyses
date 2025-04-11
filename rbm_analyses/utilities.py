from typing import ItemsView

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit


def residual_fun(
    abs_dist: np.ndarray, motor_noise: float, lr_noise: float
) -> np.ndarray:
    """This function computes updating noise (residuals) as a combination of two noise components.

    The noise variable is returned in terms of von-Mises concentration,
    which is a measure of precision, where variance = 1/concentration.

    Parameters
    ----------
    abs_dist : np.ndarray
        Absolute distance (predicted update or prediction error).
    motor_noise : np.ndarray
        Motor-noise parameter (imprecise motor control).
    lr_noise: np.ndarray
        Learning-rate-noise parameter (more noise for larger updates).

    Returns
    -------
    np.ndarray
        Updating noise expressed as von Mises concentration.

    futuretodo: add to unit tests
    """

    # Compute updating noise expressed as variance
    # (1) motor noise is updating variance due to imprecise motor control and
    # (2) learning-rate noise models more noise for larger updates
    up_noise = motor_noise + lr_noise * (np.rad2deg(abs_dist))

    # Convert std of update distribution to radians and kappa
    up_noise_radians = np.deg2rad(up_noise)
    up_var_radians = up_noise_radians**2
    kappa_up = 1 / up_var_radians

    return kappa_up


def compute_persprob(
    intercept: float, slope: float, abs_pred_up: np.ndarray
) -> np.ndarray:
    """This function computes the perseveration probability.

    Parameters
    ----------
    intercept : float
        Logistic function intercept.
    slope : float
        Logistic function slope.
    abs_pred_up : np.ndarray
        Absolute predicted update or prediction error.

    Returns
    -------
    np.ndarray
        Computed perseveration probability.
    """

    # expit(x) = 1/(1+exp(-x)), i.e., (1/(1+exp(-slope*(abs_pred_up-int))))
    return expit(slope * (abs_pred_up - intercept))


def get_sel_coeffs(items: ItemsView[str, bool], fixed_coeffs: dict, coeffs) -> dict:
    """This function extracts the model coefficients.

    Parameters
    ----------
    items : ItemsView[str, bool]
        Free parameters, specified based on which_vars dict
    fixed_coeffs : dict
        Dictionary of fixed coefficients.
    coeffs : np.ndarray
        Dictionary of free model coefficients.

    Returns
    -------
    dict
        Dictionary of selected model coefficients.
    """

    # Initialize coefficient dictionary and counters
    sel_coeffs = dict()  # initialize list with regressor names
    i = 0  # initialize counter

    # Put selected coefficients in list that is used for the regression
    for key, value in items:
        if value:
            sel_coeffs[key] = coeffs[i]
            i += 1
        else:
            sel_coeffs[key] = fixed_coeffs[key]

    return sel_coeffs


def parameter_summary(
    parameters: pd.DataFrame, param_labels: list, grid_size: tuple
) -> None:
    """This function creates a simple plot showing parameter values.

    Parameters
    ----------
    parameters : pd.DataFrame
        All parameters.
    param_labels : list
        Labels for the plot.
    grid_size : tuple
        Grid size for subplots (rows, cols)

    Returns
    -------
    None
        This function does not return any value
    """

    # Create figure
    plt.figure()

    # Cycle over parameters
    for i, label in enumerate(param_labels):

        # Create subplot
        plt.subplot(grid_size[0], grid_size[1], i + 1)
        plt.title(f"{label}")

        ax = plt.gca()
        sns.boxplot(
            y=label,
            data=parameters,
            notch=False,
            showfliers=False,
            linewidth=0.8,
            width=0.15,
            boxprops=dict(alpha=1),
            ax=ax,
            showcaps=False,
        )
        sns.swarmplot(y=label, data=parameters, color="gray", alpha=0.7, size=3, ax=ax)

        # Add y-label
        plt.ylabel(param_labels[i])

    # Adjust layout and show plot
    plt.tight_layout()
