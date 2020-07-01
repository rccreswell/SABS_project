import pints
import myokit
import numpy as np
import sabs_pkpd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy


class MyModel(pints.ForwardModel):
    def n_parameters(self):
        # Define the amount of fitted parameters
        return sabs_pkpd.constants.n

    def simulate(self, parameters, times):
        sabs_pkpd.constants.n = len(parameters)
        out = sabs_pkpd.run_model.simulate_data(parameters, sabs_pkpd.constants.s, sabs_pkpd.constants.data_exp,
                                                pre_run=sabs_pkpd.constants.pre_run)
        out = np.concatenate(out)
        return out


def parameter_is_state(param_annot, myokit_simulation):
    """"
    Returns whether the variable param_annot is a state variable of the myokit.Simulation provided as argument

    :param param_annot:
    str. name of the argument as provided in the MMT model

    :param myokit_simulation:
    myokit.Simulation. Simulation in which to look for the variable.

    :return is_state:
    bool. Returns True if the variable param_annot is a state variable, False otherwise.
    """
    # Analyse the clamped_param_annot to find component name and variable name
    i = param_annot.index('.')
    component_name = param_annot[0:i]
    variable_name = param_annot[i + 1:]

    # Load the simulation model to explore its variables
    m = myokit_simulation._model
    component = m.get(component_name, class_filter=myokit.Component)

    variable_found = False
    is_state = False

    for variable in component.variables():
        if variable.name() == variable_name:
            if variable.is_state():
                is_state = True
            variable_found = True

    if not variable_found:
        raise ValueError('The variable ' + param_annot + ' could not be found in the model.')

    return is_state


def find_index_of_state(param_annot, myokit_simulation):
    """
    Returns the index of the state vector corresponding to param_annot

    :param param_annot:
    str. name of the argument as provided in the MMT model

    :param myokit_simulation:
    myokit.Simulation. Simulation in which to look for the variable.

    :return:
    int. Returns the index of the state variable. None if the variable could not be found.
    """
    index = None
    for i in range(len(myokit_simulation._model._state)):
        if myokit_simulation._model._state[i]._component._name + '.' + myokit_simulation._model._state[i]._name ==\
                param_annot:
            index = i

    return index


def infer_params(initial_point, data_exp, boundaries_low, boundaries_high, pints_method=pints.XNES, parallel=False):
    """
    Infers parameters using PINTS library pnits.optimise() function, using method pints.XNES,and rectangular boundaries.

    :param initial_point: list
        Starting point for optimisation. It has to match the length of fitting parameters annotations.
    :param data_exp: Data_exp
        Contains the data that the model is fitted too. See documentation for sabs_pkpd.load_data for further info
    :param boundaries_low: list
        List of lower boundaries for the fitted parameters. It has to match the length of fitting parameters annotations
    :param boundaries_high: list
        List of lower boundaries for the fitted parameters. It has to match the length of fitting parameters annotations
    :return: found_parameters : numpy.array
        List of parameters values after optimisation routine.
    """

    if len(initial_point) != len(data_exp.fitting_instructions.fitted_params_annot):
        raise ValueError('The initial point should have the same length as the fitted parameters annotations' +
                         '(defined in data_exp.fitting_instructions')

    if len(boundaries_low) != len(data_exp.fitting_instructions.fitted_params_annot):
        raise ValueError('The lower boundaries should have the same length as the fitted parameters annotations' +
                         '(defined in data_exp.fitting_instructions')

    if len(boundaries_high) != len(data_exp.fitting_instructions.fitted_params_annot):
        raise ValueError('The higher boundaries should have the same length as the fitted parameters annotations' +
                         '(defined in data_exp.fitting_instructions')

    fit_values = np.concatenate(data_exp.values)

    sabs_pkpd.constants.n = len(sabs_pkpd.constants.data_exp.fitting_instructions.fitted_params_annot)

    problem = pints.SingleOutputProblem(model=MyModel(), times=np.linspace(0, 1, len(fit_values)), values=fit_values)
    boundaries = pints.RectangularBoundaries(boundaries_low, boundaries_high)
    error_measure = pints.SumOfSquaresError(problem)
    optimiser = pints.OptimisationController(error_measure, initial_point, boundaries=boundaries, method=pints_method)
    optimiser.set_parallel(parallel=parallel)
    found_parameters, found_value = optimiser.run()
    print(data_exp.fitting_instructions.fitted_params_annot)
    print(found_parameters)
    return found_parameters, found_value


def MCMC_routine(starting_point, max_iter=4000, adapt_start=None, log_prior=None,
                                mmt_model_filename=None, chain_filename = None, pdf_filename=None,
                                log_likelihood='GaussianLogLikelihood', method='HaarioBardenetACMC', sigma0=None,
                                parallel=False):
    """
    Runs a MCMC routine for the selected model

    :param starting_point:
        List of numpy.array. List of starting values for the MCMC for the optimisation parameters. Must have the same
        length as data_exp.fitting_parms_annot + 1 (for Noise). len(starting_point) defines the amount of MCMC chains.

    :param max_iter:
        int. Maximal iterations for the whole MCMC. Should be higher than adapt_start.

    :param adapt_start:
        int. Iterations before starting the adapting phase of the MCMC.

    :param log_prior: pints.log_priors
        Type of prior. If not specified, pints.UniformLogPrior

    :param mmt_model_filename: str
        location of the mmt model to run if different from the one loaded previously. It will replace the
        sabs_pkpd.constants.s myokit.Simulation() already present.

    :param chain_filename:
        str. Location of the CSV file where the chains will be written. If not provided, the chains are not saved in CSV

    :param pdf_filename:
        str. Location of the CSV file where the log_likelihood will be written. If not provided, it will not be saved
        in CSV.

    :param log_likelihood: pints.LogLikelihood
        Type of log likelihood. If not specified, pints.UnknownNoiseLogLikelihood.

    :param method: pints.method:
        method of optimisation. If not specified, pints.HaarioBardenetACMC.

    :param sigma0:
        sigma0 for the desired MCMC algorithm. If not provided, sigma0 will be computed automatically by the algorithm.
        See https://pints.readthedocs.io/en/latest/mcmc_samplers/running.html for documentation.

    :param parallel:
    Boolean. Enables or not the parallelisation of the MCMC among the available CPUs. False as default.

    :return: chains
        The chain for the MCMC routine.

    """
    sabs_pkpd.constants.n = len(starting_point[0]) -1

    if len(starting_point[0]) != len(sabs_pkpd.constants.data_exp.fitting_instructions.fitted_params_annot)+1:
        raise ValueError('Starting point and Parameters annotations + Noise must have the same length')

    if mmt_model_filename is not None:
        sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(mmt_model_filename)

    # Then create an instance of our new model class
    model = sabs_pkpd.pints_problem_def.MyModel()

    # log_prior within [0.5 * starting_point ,  2 * starting_point] if not specified
    if log_prior is not None:
        pass
    else:
        mini = np.array(np.min(starting_point, axis=0).tolist())
        maxi = np.array(np.max(starting_point, axis=0).tolist())
        log_prior = pints.UniformLogPrior(np.array(mini * 0.5).tolist(),
                                          np.array(maxi * 2).tolist())

    fit_values = np.concatenate(sabs_pkpd.constants.data_exp.values)

    problem = pints.SingleOutputProblem(model, times=np.linspace(0, 1, len(fit_values)), values=fit_values)

    # Create a log-likelihood function (adds an extra parameter!)
    log_likelihood = eval('pints.'+ log_likelihood + '(problem)')

    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    method = eval('pints.' + method)

    # Create mcmc routine
    mcmc = pints.MCMCController(log_posterior, len(starting_point), starting_point, method=method, sigma0=sigma0)

    # Allow parallelisation of computation when provided by user
    mcmc.set_parallel(parallel)

    # Add stopping criterion
    mcmc.set_max_iterations(max_iter)
    # Start adapting after adapt_start iterations
    if adapt_start is not None:
        mcmc.set_initial_phase_iterations(adapt_start)
        if adapt_start > max_iter:
            raise ValueError('The maximum number of iterations should be higher than the adapting phase length. Got ' +
                             str(max_iter) + ' maximum iterations, ' + str(adapt_start) +
                             ' iterations in adapting phase')

    if chain_filename is not None:
        mcmc.set_chain_filename(chain_filename)
    if pdf_filename is not None:
        mcmc.set_log_pdf_filename(pdf_filename)

    # Run!
    print('Running...')
    chains = mcmc.run()
    print('Done!')

    return chains


def plot_distribution_parameters(mcmc_chains: list, bound_min: list, bound_max: list, chain_index: int = 0,
                                 fig_size=(15,15), explor_iter:int =1000):
    """
    :param mcmc_chains:
    list. The list containing the MCMC chains obtained after running the MCMC routine. chain[i] returns the i-th chain.

    :param bound_min:
    list. List of length the amount of parameters sampled during the MCMC routine. bound_min[i] is the minimum boundary
    for the plot of the i-th parameter.

    :param bound_max:
    list. List of length the amount of parameters sampled during the MCMC routine. bound_max[i] is the maximum boundary
    for the plot of the i-th parameter.

    :param chain_index:
    int. Index of the chain for which the parameters distribution is plotted.

    :param fig_size:
    tuple. Defines the size of the figure on which the distribution of parameters is plotted. (15, 15) if not specified
    by the user.

    :param explor_iter:
    int. Length of the exploratory phase, which is excluded when plotting the distribution of parameters.

    :return: fig, axes
    The matplotlib.pyplot.fig and -.axes corresponding to the desired figure.
    """
    if chain_index > len(mcmc_chains)-1:
        raise ValueError('This MCMC output does not have enough chains to reach for chain no. ' + str(chain_index) +
                         '. Only ' + str(len(mcmc_chains)) + ' chains in this MCMC output.')
    # Compute the amount of graphs, given the amount of chains
    if len(mcmc_chains[0, 0]) < 4:
        n_columns = len(mcmc_chains[0, 0])
    else:
        n_columns = 4
    n_rows = 1 + len(mcmc_chains[0, 0])//4

    # Generate the subplots
    fig, axes = plt.subplots(n_rows, n_columns, figsize=fig_size)

    # Loop over the subplots
    for i in range(len(mcmc_chains[0][0])-1):
        if n_rows == 1:
            ax = axes[i]
        else:
            ax = axes[i//4, i % 4]
        hist_1d(mcmc_chains[chain_index][explor_iter:, i], ax=ax)
        ax.set_title(sabs_pkpd.constants.data_exp.fitting_instructions.fitted_params_annot[i])
        ax.set_xlim((bound_min[i], bound_max[i]))

    plt.show()

    return fig, axes


def hist_1d(x, ax):
    """
    Creates a 1d histogram and an estimate of the PDF using KDE.
    :param x : list
    PDF list that we want to plot as an histogram

    :param ax :matplotlib.axes._subplots.AxesSubplot
    Axes of the figure that we want to plot the histogram on

    :returns None
    """
    xmin = np.min(x)
    xmax = np.max(x)
    x1 = np.linspace(xmin, xmax, 100)
    x2 = np.linspace(xmin, xmax, 50)

    hist = ax.hist(x, bins=x2, density=True)
    kernel = stats.gaussian_kde(x)
    f = kernel(x1)
    ax.plot(x1, f)


def plot_MCMC_convergence(mcmc_chains, expected_values, bound_max, bound_min, parameters_annotations=None):
    """
    Plots the convergence of the MCMC chains, with boundaries and expected values.

    :param mcmc_chains:
    list. List of length number of chains, each chain having a shape:
        (number of iterations, number of parameters + 1 for Noise)

    :param expected_values:
    list. List containing the expected values of all of the parameters fitted during MCMC.

    :param bound_max:
    Maximal values allowed for each parameter

    :param bound_min:
    Minimal values allowed for each parameter

    :param parameters_annotations:
    List of strings. Names of the model parameters fitted (with noise being the last parameter) during MCMC

    :return: (fig, axes)
    """

    n_params = len(mcmc_chains[0, 0, :])

    if len(bound_min) != len(bound_max) or len(bound_min) != n_params:
        raise ValueError('Boundaries length must match the amount of parameters. Length of low boundaries: ' +
                         str(len(bound_min)) + ' , Length of upper boundaries: ' + str(len(bound_max)))

    if n_params != len(expected_values):
        raise ValueError('The expected values must have the same length as the MCMC parameters.' +
                         ' (Make sure a value is provided for noise)')

    fig_size = (12, 5 * (n_params // 2 + n_params % 2))

    fig, axes = plt.subplots(n_params // 2 + n_params % 2, 2, figsize=fig_size)

    for i in range(n_params):
        if n_params > 2:
            row = i // 2
            col = i % 2
            axes[row, col].axhline(expected_values[i], c='k', LineWidth=3)
            axes[row, col].axhline(bound_max[i], c='r', LineWidth=3)
            axes[row, col].axhline(bound_min[i], c='r', LineWidth=3)
            for j in range(len(mcmc_chains)):
                axes[row, col].plot(mcmc_chains[j, :, i], label='chain ' + str(j), LineWidth=1.5)
            axes[row, col].legend()
            if parameters_annotations is None:
                axes[row, col].set_title('Parameter ' + str(i))
            else:
                axes[row, col].set_title(parameters_annotations[i])
        else:
            col = i % 2
            axes[col].axhline(expected_values[i], c='k', LineWidth=3)
            axes[col].axhline(bound_max[i], c='r', LineWidth=3)
            axes[col].axhline(bound_min[i], c='r', LineWidth=3)
            for j in range(len(mcmc_chains)):
                axes[col].plot(mcmc_chains[j, :, i], label='chain ' + str(j), LineWidth=1.5)
            axes[col].legend()
            if parameters_annotations is None:
                axes[col].set_title('Parameter ' + str(i))
            else:
                axes[col].set_title(parameters_annotations[i])

    return fig, axes
