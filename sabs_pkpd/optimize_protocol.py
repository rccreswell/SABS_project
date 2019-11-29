"""Optimize the input protocol using the Fisher Information method of [1]_.

(In Progress)
# TODO: generalize model and protocol
# TODO: better optimization algorithm for protocol parameters?
# TODO: speed up the code

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


def output(a, b, amplitude, duration):
    """Generate the time-series given model parameters and protocol parameters.

    Parameters
    ----------
    a : float
        growth rate
    b : float
        self interaction
    amplitude : float
        magnitude of step protocol
    duration : float
        duration of step protocol

    Returns
    -------
    np.array
        output value over time
    """
    # Must constain model parameters, the derivative is sometimes trying to use
    # values which lead to unpleasant dynamics
    b_use = 0 if b > 0 else b
    a_use = 0 if a < 0 else a

    # Function for the protocol over time
    protocol = lambda times : \
             ((times > 1.0) & (times < 1.0 + duration))#.astype(float)

    # Function for do/dt
    def f(t, x):
        return a_use*x + b_use*x**2 + amplitude * protocol(t)

    t = np.linspace(0, 10, 1000)
    result = scipy.integrate.solve_ivp(
                f, (0,10), [10], t_eval=t, max_step=0.1, vectorized=True).y[0]

    return result


def objective(amplitude, duration):
    """The objective for protocol optimization, obtained via Fisher information.

    Parameters
    ----------
    amplitude : float
        magnitude of step protocol
    duration : float
        duration of step protocol

    Returns
    -------
    float
        Current value of the objective
    """
    true_sigma = 0.1
    true_a = 1.0
    true_b = -0.1

    output_a = lambda a : output(a, true_b, amplitude, duration)
    output_b = lambda b : output(true_a, b, amplitude, duration)

    dfda = nd.Derivative(output_a, n=1)
    dfda = dfda(true_a)

    dfdb = nd.Derivative(output_b, n=1)
    dfdb = dfdb(true_b)

    all_derivs = [dfda, dfdb]
    J = np.zeros((2,2))
    for i, d1 in enumerate(all_derivs):
        for j, d2 in enumerate(all_derivs):
            J[i,j] = np.sum(d1 * d2)

    J = 1/true_sigma**2 * J
    J_inv = np.linalg.inv(J)

    true_params = [true_a, true_b]
    F = np.sum([J_inv[i,i] / true_params[i] for i in range(len(true_params))])

    return F


class MyModel(pints.ForwardModel):
    def n_parameters(self):
        return 2

    def simulate(self, parameters, times):
        return output(parameters[0],
                      parameters[1],
                      self.amplitude,
                      self.duration)


def learn_parameters(amplitude, duration):
    """Find posterior of the model parameters.
    """
    times = np.linspace(0, 10, 1000)
    values = output(1.0, -0.1, amplitude, duration)
    values += np.random.normal(0, 0.1, len(values))
    m = MyModel()
    m.amplitude = amplitude
    m.duration = duration
    problem = pints.SingleOutputProblem(m, times, values)
    likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 0.1)
    prior = pints.UniformLogPrior([0.0, -10.0], [10.0, 0.0])
    log_posterior = pints.LogPosterior(likelihood, prior)
    x0 = [[1, -0.1]]
    mcmc = pints.MCMCController(log_posterior, 1, x0)
    mcmc.set_max_iterations(4000)
    chains = mcmc.run()
    return chains


def protocol_traj(times, amplitude, duration):
    return np.array(((times > 1.0) & (times < 1.0 + duration))).astype(float) * amplitude


if __name__ == '__main__':
    np_objective = lambda protocol_params : objective(protocol_params[0], protocol_params[1])

    times = np.linspace(0, 10, 1000)

    fig = plt.figure()
    ax1 = plt.subplot(2, 4, 1)

    # Initial protocol
    protocol = (0.5, 1.5)
    p0 = protocol_traj(times, *protocol)
    ax1.plot(times, p0)
    ax1.set_title('initial protocol')

    # Values from that protocol
    values = output(1.0, -0.1, *protocol)
    values += np.random.normal(0, 0.1, len(values))
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(times, values)
    ax2.set_title('data')

    p = learn_parameters(*protocol)
    alphas = p[0,:,0][2000:]
    betas = p[0,:,1][2000:]
    ax3 = plt.subplot(2, 4, 3)
    ax3.hist(alphas, alpha=0.5)
    ax3.set_title('alpha')

    ax4 = plt.subplot(2, 4, 4)
    ax4.hist(betas, alpha=0.5)
    ax4.set_title('beta')


    res = scipy.optimize.minimize(np_objective, x0=protocol, method='L-BFGS-B', options={'eps':5e-2, 'disp':True, 'maxiter':100})
    protocol = res.x
    # protocol = [9.56993054, 0.50000546]
    print('optimized_protocol', protocol)

    ax5 = plt.subplot(2, 4, 5, sharey=ax1)
    p1 = protocol_traj(times, *protocol)
    ax5.plot(times, p1)
    ax5.set_title('optimized protocol')
    ax5.set_xlabel('time')

    # Values from that protocol
    values = output(1.0, -0.1, *protocol)
    values += np.random.normal(0, 0.1, len(values))
    ax6 = plt.subplot(2, 4, 6, sharey=ax2)
    ax6.plot(times, values)
    ax6.set_title('data')
    ax6.set_xlabel('time')

    p = learn_parameters(*protocol)
    alphas = p[0,:,0][2000:]
    betas = p[0,:,1][2000:]
    ax7 = plt.subplot(2, 4, 7, sharex=ax3)
    ax7.hist(alphas, alpha=0.5)
    ax7.set_title('alpha')

    ax8 = plt.subplot(2, 4, 8, sharex=ax4)
    ax8.hist(betas, alpha=0.5)
    ax8.set_title('beta')

    fig.set_tight_layout(True)
    plt.show()



# error_measure = pints.SumOfSquaresError(problem)
# initial_points = [1, -0.1]
# opt = pints.OptimisationController(error_measure, initial_points)
# parameters, error = opt.run()
# return parameters
