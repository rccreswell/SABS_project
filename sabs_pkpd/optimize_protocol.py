"""Optimize the input protocol using the Fisher Information method [1]_.

References
----------
.. [1] Alexander, Daniel C. "A general framework for experiment design in
       diffusion MRI and its application in measuring direct tissue-
       microstructure features." Magnetic Resonance in Medicine 60.2 (2008):
       439-448.
"""

import scipy.integrate
import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt
import pints
import pints.plot
from multiprocessing import Pool
import os
import pyswarms as ps


class ProtocolOptimizer:
    """Handles optimization of the protocol for parameter inference.

    This object is used to hold all the necessary pieces of protocol
    optimization, and run the optimization.
    """
    def __init__(self,
                 simulator,
                 protocol_form,
                 times,
                 x0,
                 model_params,
                 protocol_params,
                 num_mcmc_iterations,
                 true_model_params=None,
                 parallel=True):
        """
        Parameters
        ----------
        simulator : function
            This function should accept model parameters, the protocol as a
            function of time, a set of time points, and an initial condition,
            and return the output over time. It will be called like this:
            f(*model_params, protocol, times, x0)
            where model_params is a list giving the current values of the model
            parameters, and protocol is the function returned by protocol_form
        protocol_form : function
            This function accepts certain parameters and returns a function of
            time giving the protocol over time. The parameters of this function
            are what will be optimized by the algorithm.
        times : np.ndarray
            The grid of time points for which to evaluate the output
        x0 : float
            The value of x at time=0
        model_params : list of float
            The initial values of the model parameters. Must be
            compatible with simulator. These are used to evaluate the objective
            function.
        protocol_params : list of float
            The initial values of the protocol parameters. Must be compatible
            with protocol_form
        num_mcmc_iterations : int
            The number of MCMC iterations to use when sampling posteriors.
        true_model_params : list of float, optional
            The ground truth model parameters. If not supplied, set equal to
            model_params. These are used for synthetic data generation and
            plotting.
        """
        self.times = times
        self.x0 = x0
        self.simulator = simulator
        self.protocol_form = protocol_form
        self.model_params = model_params
        self.protocol_params = protocol_params

        self.num_mcmc_iterations = num_mcmc_iterations

        if true_model_params is not None:
            self.true_model_params = true_model_params
        else:
            self.true_model_params = self.model_params.copy()

        self.parallel = parallel

    def run_original_protocol(self):
        """Save the initial protocol parameter values and resulting posteriors.

        Call this function before optimizing parameters, to see what the
        performance was with the starting protocol.
        """
        # Save the current protocol parameters
        self.original_protocol_params = self.protocol_params.copy()

        # Run Bayesian inference
        self.infer_model_parameters()

        # Save the synthetic data and posterior distributions
        self.original_data = self.data.copy()
        self.original_posterior = self.posterior.copy()

    def objective(self, protocol_params):
        """Evaluate the protocol optimization objective function.

        Parameters
        ----------
        protocol_params : list of float
            The protocol parameters at which to evaluate the objective function

        Returns
        -------
        float
            The value of the objective function
        """
        # If a matrix of protocol params is supplied, evaluate each one
        # in parallel and return a list of objectives
        if type(protocol_params) is np.ndarray:
            if protocol_params.ndim == 2:
                if self.parallel:
                    pool = Pool(os.cpu_count())
                    all_objectives = pool.map(self.objective,
                                              [p for p in protocol_params])

                    pool.close()
                    pool.join()

                    return np.array(all_objectives)

                else:
                    all_objectives = []
                    for p in [p for p in protocol_params]:
                        all_objectives.append(self.objective(p))
                    return np.array(all_objectives)

        # Otherwise, evaluate the objective for one protocol params

        # Get the protocol as a function of t
        protocol = self.protocol_form(*protocol_params)

        # Get output(t) as a function of each model variable
        partials = []
        for i, param in enumerate(self.model_params):
            def partial(x):
                return self.simulator(
                    *[p if j != i else x for j, p in
                        enumerate(self.model_params)],
                    protocol,
                    self.times,
                    self.x0)

            partials.append(partial)

        # Take the derivatives with respect to each model variable
        derivatives = []
        for i, partial in enumerate(partials):
            dfda = nd.Derivative(partial, n=1, step=1e-2)

            # Evaluate the derivative at that model parameter value
            dfda = dfda(self.model_params[i])
            derivatives.append(dfda)

        # Form the Fisher Information Matrix
        J = np.zeros((len(derivatives), len(derivatives)))
        for i, d1 in enumerate(derivatives):
            for j, d2 in enumerate(derivatives):
                J[i, j] = np.sum(d1 * d2)

        true_sigma = 0.1
        J = 1 / true_sigma ** 2 * J

        # Calculate the objective function from the CRLBs
        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            # Singular matrix when all zeros, so changing the parameters has no
            # effect on the trajectory. This indicates a very bad choice so
            # manually insert a high value for the objective
            J_inv = np.diag([1e50] * len(self.model_params))

        # Sum the values along the diagonal, scaled by model parameter value
        F = np.sum([J_inv[i, i] / self.model_params[i]
                    for i in range(len(self.model_params))])

        return F

    def optimize_protocol(self, max_iters=100):
        """Optimize the protocol parameters to maximize the Fisher objective.
        """
        def objective(protocol_params):
            return self.objective(protocol_params)

        num_particles = 25
        options = {'c1': 0.9, 'c2': 0.2, 'w': 0.9}
        dim = len(self.protocol_params)

        # Start one particle at the baseline protocol. Start all the others
        # at randomly generated points about it
        init_pos = [self.protocol_params]
        for _ in range(num_particles - 1):
            init_pos.append(np.random.normal(self.protocol_params,
                                             np.abs(self.protocol_params) * 5))
        init_pos = np.array(init_pos)

        optimizer = ps.single.GlobalBestPSO(n_particles=num_particles,
                                            dimensions=dim,
                                            options=options,
                                            init_pos=init_pos)

        cost, pos = optimizer.optimize(objective, iters=max_iters)
        protocol = pos
        self.protocol_params = protocol

    def infer_model_parameters(self):
        """Infer posteriors of the model parameters given the current protocol.
        """
        values, chains = infer_model_parameters(
            self.simulator,
            self.x0,
            self.protocol_form(*self.protocol_params),
            self.true_model_params,
            self.times,
            self.num_mcmc_iterations
        )

        self.data = values
        self.posterior = chains

    def update_model_parameters(self):
        """Set the model parameters to the current posterior medians.

        Requires that self.posterior be populated with the desired MCMC
        samples, by running self.infer_model_parameters() first.
        """
        burnin = self.num_mcmc_iterations // 2

        for i in range(len(self.model_params)):
            posterior = self.posterior[0, :, i][burnin:]
            self.model_params[i] = np.median(posterior)

    def plot(self, show=True):
        """Make a plot of the original and optimized protocols.
        """
        burnin = self.num_mcmc_iterations // 2

        # columns = protocol, data, and one posterior per model parameter
        n_cols = 2 + len(self.model_params)

        fig, axes = plt.subplots(2, n_cols, sharex='col', sharey='col')

        # Initial protocol
        axes[0, 0].plot(
            self.times,
            self.protocol_form(*self.original_protocol_params)(self.times)
        )
        axes[0, 0].set_title('initial protocol')

        axes[0, 1].plot(self.times, self.original_data)
        axes[0, 1].set_title('data from initial protocol')

        for i, model_param in enumerate(self.model_params):
            true_value = self.true_model_params[i]
            axes[0, i + 2].hist(
                self.original_posterior[0, :, i][burnin:], alpha=0.8)
            axes[0, i + 2].axvline(
                true_value, zorder=-10, color='mediumseagreen')
            axes[0, i + 2].set_title('parameter {}'.format(i + 1))

        # Optimized protocol
        axes[1, 0].plot(self.times,
                        self.protocol_form(*self.protocol_params)(self.times))
        axes[1, 0].set_title('optimized protocol')

        axes[1, 1].plot(self.times, self.data)
        axes[1, 1].set_title('data from optimized protocol')

        for i, model_param in enumerate(self.model_params):
            true_value = self.true_model_params[i]
            axes[1, i + 2].hist(self.posterior[0, :, i][burnin:], alpha=0.8)
            axes[1, i + 2].axvline(
                true_value, zorder=-10, color='mediumseagreen')
            axes[1, i + 2].set_title('parameter {}'.format(i + 1))

        fig.set_tight_layout(True)
        if show:
            plt.show()
        else:
            return fig


def infer_model_parameters(simulator,
                           xinit,
                           protocol,
                           true_model_params,
                           times,
                           num_mcmc_iterations):
    """Infer model parameters from synthetic data.

    Parameters
    ----------
    simulator : function
        See ProtocolOptimizer documentation
    xinit : list of float
        x at time=0
    protocol : function
        function of time giving the protocol over time
    true_model_params : list of float
        The ground truth model parameters from which to generate synthetic data
    times : np.ndarray
        The time points for evaluating the output

    Returns
    -------
    np.ndarray
        The synthetic data time-series
    np.ndarray
        MCMC chains of the posterior
    """
    class MyModel(pints.ForwardModel):
        def n_parameters(self):
            return len(true_model_params)

        def simulate(self, parameters, times):
            return simulator(*parameters, protocol, times, xinit)

    # Generate the synthetic dataset
    values = simulator(*true_model_params, protocol, times, xinit)

    # Add noise
    values += np.random.normal(0, 0.1, len(values))

    # Pints bayesian inference
    m = MyModel()
    problem = pints.SingleOutputProblem(m, times, values)
    likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 0.1)

    prior = pints.UniformLogPrior([0.0] * len(true_model_params),
                                  [100.0] * len(true_model_params))

    log_posterior = pints.LogPosterior(likelihood, prior)

    # Get the MCMC starting point near the true values
    true_model_params = np.array(true_model_params)
    x0 = true_model_params + np.random.normal(
        0, np.abs(true_model_params) * 0.1, len(true_model_params))
    x0 = [x0]

    # Run MCMC chain
    mcmc = pints.MCMCController(log_posterior, 1, x0)
    mcmc.set_max_iterations(num_mcmc_iterations)
    chains = mcmc.run()

    return values, chains


def damped_harmonic_oscillator(c, k, m, beta, protocol, t, x0):
    """Damped harmonic oscillator with forcing.

    d^2x/dt^2 + c/m dx/dt + k/m x = 1/m protocol(t)

    The solution is accomplished by converting this second-order equation into
    a system of two first-order ODEs. Only the solution for x is returned.

    Parameters
    ----------
    c : float
        damping constant
    k : float
        spring constant
    m : float
        mass
    """
    m = 1e-1 if m < 0 else m
    c = 0 if c < 0 else c
    k = 0 if k < 0 else k

    def f(t, x):
        d = [0, 0]
        d[0] = x[1]
        d[1] = 1 / m * protocol(t) - c / m * x[1] - k / m * x[0] \
            - beta / m * x[0] ** 3
        return d

    result = scipy.integrate.solve_ivp(
        f,
        (min(t), max(t)),
        [x0, 0],
        t_eval=t,
        max_step=0.1,
        vectorized=True).y[0]

    return result


# TODO: remove this function, it is now in Examples
# def main():
#     params = [1.70015678e+00,
#               1.92681600e+00,
#               -1.12329275e-01,
#               9.86565896e-01,
#               4.02746746e-01,
#               1.96004128e+02,
#               7.16528663e+01,
#               -2.13426171e+01,
#               1.32499357e+01,
#               -4.06159185e+01]
#
#     opt = ProtocolOptimizer(
#         damped_harmonic_oscillator,
#         five_event_protocol,
#         np.linspace(0, 10, 400),
#         0.0,
#         [5.0, 10.0, 1.0, 10.0],
#         params,
#         4000,
#         true_model_params=[5.0, 10.0, 1.0, 10.0])
#
#     opt.run_original_protocol()
#     opt.optimize_protocol()
#     opt.infer_model_parameters()
#     opt.plot()
#     opt.update_model_parameters()


# if __name__ == '__main__':
#     main()
