import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns


def residual_fun(abs_dist, motor_noise, lr_noise):
    """ This function computes updating noise (residuals) as a combination of two noise components

    The noise variable is returned in terms of von-Mises concentration,
    which is a measure of precision, where variance = 1/concentration.

    :param abs_dist: Absolute distance (predicted update or prediction error)
    :param motor_noise: Motor-noise parameter (imprecise motor control)
    :param lr_noise: Learning-rate-noise parameter (more noise for larger updates)
    :return: kappa: Updating noise expressed as von Mises concentration.

    # futuretodo: add to unit tests
    """

    # Compute updating noise expressed as variance
    # (1) motor noise is updating variance due to imprecise motor control and
    # (2) learning-rate noise models more noise for larger updates
    up_noise = motor_noise + lr_noise * (np.rad2deg(abs_dist))

    # Convert std of update distribution to radians and kappa
    up_noise_radians = np.deg2rad(up_noise)
    up_var_radians = up_noise_radians ** 2
    kappa_up = 1 / up_var_radians

    return kappa_up


def compute_persprob(intercept, slope, abs_pred_up):
    """ This function computes the perseveration probability

    :param intercept: Logistic function intercept
    :param slope: Logistic function slope
    :param abs_pred_up: Absolute predicted update
    :return: Computed perseveration probability
    """

    # expit(x) = 1/(1+exp(-x)), i.e., (1/(1+exp(-slope*(abs_pred_up-int))))
    return expit(slope * (abs_pred_up - intercept))


def get_sel_coeffs(items, fixed_coeffs, coeffs):
    """ This function extracts the model coefficients

    :param items: 
    :param fixed_coeffs:
    :param coeffs:
    :return:
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


def parameter_summary(parameters, param_labels, grid_size):
    """ This function creates a simple plot showing parameter values

    :param parameters: All parameters
    :param param_labels: Labels for the plot
    :param grid_size: Grid size for subplots (rows, cols)
    :return: None
    """

    # Create figure
    plt.figure()

    # Cycle over parameters
    for i, label in enumerate(param_labels):

        # Create subplot
        plt.subplot(grid_size[0], grid_size[1], i + 1)
        plt.title(f"{label}")

        ax = plt.gca()
        sns.boxplot(y=label, data=parameters,
                    notch=False, showfliers=False, linewidth=0.8, width=0.15,
                    boxprops=dict(alpha=1), ax=ax, showcaps=False)
        sns.swarmplot(y=label, data=parameters, color='gray', alpha=0.7, size=3, ax=ax)

        # Add y-label
        plt.ylabel(param_labels[i])

    # Adjust layout and show plot
    plt.tight_layout()
