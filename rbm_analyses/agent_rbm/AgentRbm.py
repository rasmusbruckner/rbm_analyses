"""Agent: Implementation of the reduced Bayesian model"""

import sys

import numpy as np
from all_in import safe_div
from scipy.stats import norm


class AlAgent:
    """This class definition specifies the properties of the object that implements the reduced Bayesian model

    The model infers the mean of the outcome-generating distribution according to change-point probability and
    relative uncertainty.
    """

    def __init__(self, agent_vars):
        # This function creates an agent object of class Agent based on the agent initialization input

        # Set variable task properties based on input
        self.s = agent_vars.s
        self.h = agent_vars.h
        self.u = agent_vars.u
        self.q = agent_vars.q
        self.sigma = agent_vars.sigma
        self.sigma_t_sq = agent_vars.sigma_0
        self.sigma_H = agent_vars.sigma_H
        self.tau_t = agent_vars.tau_0
        self.omega_t = agent_vars.omega_0
        self.mu_t = agent_vars.mu_0
        self.max_x = agent_vars.max_x
        self.circular = agent_vars.circular

        # Initialize variables
        self.a_t = np.nan  # belief update
        self.alpha_t = np.nan  # learning rate
        self.tot_var = np.nan  # total uncertainty
        self.C = np.nan  # term related to catch-trial helicopter cue

    # Futuretodo: Create sub-function as in sampling agent
    def learn(
        self, delta_t: float, b_t: float, v_t: int, mu_H: int, high_val: int
    ) -> None:
        """This function implements the inference of the reduced Bayesian model.

        Parameters
        ----------
        delta_t : float
            Current prediction error
        b_t : float
            Last prediction of participant
        v_t : int
            Helicopter visibility
        mu_H :
            True helicopter location
        high_val :
            High-value index

        Returns
        -------
        None
            This function does not return any value.

        futuretodo:
        use "mypy" typechecker
        use getters
        """

        if np.isnan(delta_t):
            # Ensure that delta is not NaN so that model is not accidentally applied to wrong data
            sys.exit("delta_t is NaN")

        # Update variance of predictive distribution
        self.tot_var = self.sigma**2 + self.sigma_t_sq

        # Compute changepoint probability
        # -------------------------------

        # Likelihood of prediction error given that change point occurred: (1/max_x)^s * h
        term_1 = ((1 / self.max_x) ** self.s) * self.h

        # Likelihood of prediction error given that no changepoint occurred:
        # (N(delta_t; 0,sigma^2_t + sigma^2))^s * (1-h)
        term_2 = (norm.pdf(delta_t, 0, np.sqrt(self.tot_var)) ** self.s) * (1 - self.h)

        # Compute change-point probability
        self.omega_t = safe_div(term_1, (term_2 + term_1))  # eq. 8

        # Compute learning rate and update belief
        # ---------------------------------------
        self.alpha_t = self.omega_t + self.tau_t - self.tau_t * self.omega_t

        # Add reward bias to learning rate and correct for learning rates > 1 and < 0
        self.alpha_t = self.alpha_t + self.q * high_val
        if self.alpha_t > 1.0:
            self.alpha_t = 1.0
        elif self.alpha_t < 0.0:
            self.alpha_t = 0.0

        # Set model belief equal to last prediction of participant to estimate model using subjective prediction errors
        self.mu_t = b_t

        # hat{a_t} := alpha_t * delta_t
        self.a_t = self.alpha_t * delta_t

        # mu_{t+1} := mu_t + hat{a_t}
        if not self.circular:
            self.mu_t = self.mu_t + self.a_t
        else:
            # Wrap mu_t around circle
            self.mu_t = self.mu_t % self.max_x

        # On catch trials, take true mean into consideration
        # --------------------------------------------------
        if v_t:
            # Compute weight of true mean
            # w_t := sigma_t^2 / (sigma_t^2 + sigma_H^2)
            w_t = self.sigma_t_sq / (self.sigma_t_sq + self.sigma_H**2)

            # Compute mean of inferred distribution with additional mean information
            # mu_t = (1 - w_t) * mu_{t+1} + w_t * mu_H
            self.mu_t = (1 - w_t) * self.mu_t + w_t * mu_H

            # Recompute the model's update under consideration of the catch-trial information
            # \hat{a}_t = mu_{t+1} - b_t, same as in data preprocessing
            self.a_t = self.mu_t - b_t

            # Compute mixture variance of the two distributions...
            # C := 1 / ((1/sigma_t^2) + (1/sigma_H^2))
            term_1 = safe_div(1, self.sigma_t_sq)
            term_2 = safe_div(1, self.sigma_H**2)
            self.C = safe_div(1, term_1 + term_2)

            # ...and update relative uncertainty accordingly
            # tau_t = C / (C + sigma^2)
            self.tau_t = safe_div(self.C, self.C + self.sigma**2)

        # Update relative uncertainty of the next trial
        # ---------------------------------------------

        # Update estimation uncertainty:
        # sigma_{t+1}^2 := (omega_t * sigma^2
        #                   + (1-omega_t) * tau_t * sigma^2
        #                   + omega_t * (1 - omega_t) * (delta_t * (1 - tau_t))^2) / exp(u)

        # Note that u is already in exponential form
        term_1 = self.omega_t * (self.sigma**2)
        term_2 = (1 - self.omega_t) * self.tau_t * (self.sigma**2)
        term_3 = self.omega_t * (1 - self.omega_t) * ((delta_t * (1 - self.tau_t)) ** 2)
        self.sigma_t_sq = safe_div((term_1 + term_2 + term_3), self.u)  # eq. 9

        # Update relative uncertainty:
        # tau_{t+1} := sigma_{t+1}^2 / (sigma_{t+1}^2 + sigma^2)
        self.tau_t = safe_div(
            self.sigma_t_sq, (self.sigma_t_sq + self.sigma**2)
        )  # eq. 10

    # @property
    # def omega_t(self):
    #     """Get the current omega_t."""
    #     return self.omega_t
