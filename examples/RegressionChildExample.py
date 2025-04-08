"""This class is an example of a child class for a circular regression analysis"""

import numpy as np
import pandas as pd
from rbm_analyses import RegressionParent


class RegressionChildExample(RegressionParent):
    """This class specifies the instance variables and methods of the example of the regression analysis"""

    def __init__(self, reg_vars):
        """This function defines the instance variables unique to each instance

            See RegVarsExample for documentation

        :param reg_vars: Regression-variables-object instance
        """

        # Parameters from parent class
        super().__init__(reg_vars)

        # Extract parameter names for data frame
        self.beta_0 = reg_vars.beta_0
        self.beta_1 = reg_vars.beta_1
        self.omikron_0 = reg_vars.omikron_0
        self.omikron_1 = reg_vars.omikron_1
        self.lambda_0 = reg_vars.lambda_0
        self.lambda_1 = reg_vars.lambda_1

        # Extract staring points
        self.beta_0_x0 = reg_vars.beta_0_x0
        self.beta_1_x0 = reg_vars.beta_1_x0
        self.omikron_0_x0 = reg_vars.omikron_0_x0
        self.omikron_1_x0 = reg_vars.omikron_1_x0
        self.lambda_0_x0 = reg_vars.lambda_0_x0
        self.lambda_1_x0 = reg_vars.lambda_1_x0

        # Extract range of random starting point values
        self.beta_0_x0_range = reg_vars.beta_0_x0_range
        self.beta_1_x0_range = reg_vars.beta_1_x0_range
        self.omikron_0_x0_range = reg_vars.omikron_0_x0_range
        self.omikron_1_x0_range = reg_vars.omikron_1_x0_range
        self.lambda_0_x0_range = reg_vars.lambda_0_x0_range
        self.lambda_1_x0_range = reg_vars.lambda_1_x0_range

        # Extract boundaries for estimation
        self.beta_0_bnds = reg_vars.beta_0_bnds
        self.beta_1_bnds = reg_vars.beta_1_bnds
        self.omikron_0_bnds = reg_vars.omikron_0_bnds
        self.omikron_1_bnds = reg_vars.omikron_1_bnds
        self.lambda_0_bnds = reg_vars.lambda_0_bnds
        self.lambda_1_bnds = reg_vars.lambda_1_bnds

        # Extract free parameters
        self.which_vars = reg_vars.which_vars

        # Extract fixed parameter values
        self.fixed_coeffs_reg = reg_vars.fixed_coeffs_reg

    @staticmethod
    def get_datamat(df):
        """This function creates the explanatory matrix

        :param df: Data frame containing subset of data
        :return: reg_df: Regression data frame
        """

        # Create custom data matrix for project
        reg_df = pd.DataFrame(columns=["delta_t"])
        reg_df["int"] = np.ones(len(df))
        reg_df["delta_t"] = df["delta_t"].to_numpy()
        reg_df["a_t"] = df["a_t"].to_numpy()
        reg_df["group"] = df["group"].to_numpy()

        return reg_df

    def get_starting_point(self):
        """This function determines the starting points of the estimation process

        :return: x0: List with starting points
        """

        # Put all starting points into list
        if self.rand_sp:

            # Draw random starting points
            x0 = [
                np.random.uniform(self.beta_0_x0_range[0], self.beta_0_x0_range[1]),
                np.random.uniform(self.beta_1_x0_range[0], self.beta_1_x0_range[1]),
                np.random.uniform(
                    self.omikron_0_x0_range[0], self.omikron_0_x0_range[1]
                ),
                np.random.uniform(
                    self.omikron_1_x0_range[0], self.omikron_1_x0_range[1]
                ),
            ]

        else:

            # Use fixed starting points
            x0 = [self.beta_0_x0, self.beta_1_x0, self.omikron_0_x0, self.omikron_1_x0]

        return x0
