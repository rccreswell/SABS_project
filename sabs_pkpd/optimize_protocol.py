"""Optimize the input protocol using the Fisher Information method of [1]_.

(In Progress)
# TODO: generalize noise (currently it is Gaussian with known, fixed sigma)
# TODO: add model parameter priors
# TODO: better optimization algorithm for protocol parameters?
# TODO: speed up the code

Current structure of the code:

The ProtocolOptimizer object should now handle general models and protocols.
The user must supply two functions when constructing the ProtocolOptimizer
which specify the model and the form of the protocol. See the documentation
there.

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


class ProtocolOptimizer:
    """Handles optimization of the protocol for parameter inference.
    """
    def __init__(self,
                 simulator,
                 protocol_form,
                 times,
                 x0,
                 model_params,
                 protocol_params):
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
            The grid of time-points for which to evaluate the output
        x0 : float
            The value x(time=0)
        model_params : list of float
            The initial values of the model parameters. Must be compatible with
            simulator
        protocol_params : list of float
            The initial values of the protocol parameters. Must be compatible
            with protocol_form
        """
        self.times = times
        self.x0 = x0
        self.simulator = simulator
        self.protocol_form = protocol_form
        self.model_params = model_params
        self.protocol_params = protocol_params


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
        # Get the protocol as a function of t
        protocol = self.protocol_form(*protocol_params)

        # Get output(t) as a function of each model variable
        partials = []
        for i, param in enumerate(self.model_params):
            partial = lambda x: self.simulator(
                *[p if j != i else x for j, p in enumerate(self.model_params)],
                protocol,
                self.times,
                self.x0)

            partials.append(partial)

        # Take the derivatives with respect to each model variable
        derivatives = []
        for i, partial in enumerate(partials):
            dfda = nd.Derivative(partial, n=1)

            # Evaluate the derivative at that model parameter value
            dfda = dfda(self.model_params[i])
            derivatives.append(dfda)

        # Form the Fisher Information Matrix
        J = np.zeros((len(derivatives), len(derivatives)))
        for i, d1 in enumerate(derivatives):
            for j, d2 in enumerate(derivatives):
                J[i,j] = np.sum(d1 * d2)

        true_sigma = 0.1
        J = 1/true_sigma**2 * J

        # Calculate the objective function from the CRLBs
        J_inv = np.linalg.inv(J)
        F = np.sum([J_inv[i,i] / self.model_params[i] \
                    for i in range(len(self.model_params))])

        return F


    def optimize_protocol(self):
        """Optimize the protocol parameters to maximize the Fisher objective.
        """
        objective = lambda protocol_params : self.objective(protocol_params)

        res = scipy.optimize.minimize(
                  objective,
                  x0=self.protocol_params,
                  method='L-BFGS-B',
                  options={'eps':5e-2, 'disp':True, 'maxiter':100}
              )

        protocol = res.x
        print('optimized_protocol', protocol)
        self.protocol_params = protocol


    def infer_model_parameters(self):
        """Infer posteriors of the model parameters given the current protocol.
        """
        values, chains = infer_model_parameters(
                            self.simulator,
                            self.x0,
                            self.protocol_form(*self.protocol_params),
                            self.model_params,
                            self.times
                         )

        self.data = values
        self.posterior = chains


    def plot(self):
        """Make a plot of the original and optimized protocols.
        """
        burnin = 2000

        # columns = protocol, data, and one posterior per model parameter
        n_cols = 2 + len(self.model_params)

        fig, axes = plt.subplots(2, n_cols, sharex='col', sharey='col')

        # Initial protocol
        axes[0,0].plot(self.times,
                self.protocol_form(*self.original_protocol_params)(self.times))
        axes[0,0].set_title('initial protocol')

        axes[0,1].plot(self.times, self.original_data)
        axes[0,1].set_title('data from initial protocol')

        for i, model_param in enumerate(self.model_params):
            axes[0,i+2].hist(self.original_posterior[0,:,i][burnin:])
            axes[0,i+2].set_title('parameter {}'.format(i+1))

        # Optimized protocol
        axes[1,0].plot(self.times,
                        self.protocol_form(*self.protocol_params)(self.times))
        axes[1,0].set_title('optimized protocol')

        axes[1,1].plot(self.times, self.data)
        axes[1,1].set_title('data from optimized protocol')

        for i, model_param in enumerate(self.model_params):
            axes[1,i+2].hist(self.posterior[0,:,i][burnin:])
            axes[1,i+2].set_title('parameter {}'.format(i+1))

        fig.set_tight_layout(True)
        plt.show()


def infer_model_parameters(simulator,
                           xinit,
                           protocol,
                           true_model_params,
                           times):
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

    prior = pints.UniformLogPrior([-100.0]*len(true_model_params),
                                  [100.0]*len(true_model_params))

    log_posterior = pints.LogPosterior(likelihood, prior)

    # Get the MCMC starting point near the true values
    true_model_params = np.array(true_model_params)
    x0 = true_model_params + np.random.normal(
                    0, np.abs(true_model_params)*0.1, len(true_model_params))
    x0 = [x0]

    # Run MCMC chain
    mcmc = pints.MCMCController(log_posterior, 1, x0)
    mcmc.set_max_iterations(4000)
    chains = mcmc.run()

    return values, chains


##### Below = My protocols and models for testing purposes #####

def one_step_protocol(amplitude, duration):
    return lambda times : np.array(((times > 1.0) & (times < 1.0 + duration))).astype(float) * amplitude


def logistic_growth_additive_protocol(alpha, beta, protocol, t, x0):
    """Logistic growth model with additive stimulus.

    dx/dt = alpha * x + beta * x^2 + protocol(t)

    Parameters
    ----------
    alpha : float
        Growth rate
    beta : float
        Self limiting term
    protocol : function of time
        input stimulus
    t : numpy.ndarray
        time points
    x0 : float
        x at time=0

    Returns
    -------
    numpy.ndarray
        time series
    """
    # Must constain model parameters, the derivative is sometimes trying to use
    # values which lead to unpleasant dynamics
    alpha = 0 if alpha < 0 else alpha
    beta = 0 if beta > 0 else beta

    def f(t, x):
        return alpha*x + beta*x**2 + protocol(t)

    result = scipy.integrate.solve_ivp(
                f, (min(t),max(t)), [x0], t_eval=t, max_step=0.1, vectorized=True).y[0]

    return result


def main():
    opt = ProtocolOptimizer(logistic_growth_additive_protocol,
                           one_step_protocol,
                           np.linspace(0, 10, 1000),
                           10,
                           [1, -0.1],
                           [0.5, 1.5])

    opt.run_original_protocol()
    opt.optimize_protocol()
    opt.infer_model_parameters()
    opt.plot()


if __name__ == '__main__':
    main()