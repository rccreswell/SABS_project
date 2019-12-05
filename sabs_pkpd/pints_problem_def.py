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

        out = sabs_pkpd.run_model.simulate_data(parameters, sabs_pkpd.constants.s, sabs_pkpd.constants.data_exp, pre_run = sabs_pkpd.constants.pre_run)
        out = np.concatenate([i for i in out])
        return out


def infer_params(initial_point, data_exp, boundaries_low, boundaries_high):
    """
    Infers parameters using PINTS library pnits.optimise() function, using method pints.XNES,and rectangular boundaries.

    :param initial_point: list
        Starting point for optimisation. It has to match the length of fitting parameters annotations.
    :param data_exp: Data_exp
        Contains the data that the model is fitted too. See documentation for sabs_pkpd.load_data for further information.
    :param boundaries_low: list
        List of lower boundaries for the fitted parameters. It has to match the length of fitting parameters annotations.
    :param boundaries_high: list
        List of lower boundaries for the fitted parameters. It has to match the length of fitting parameters annotations.
    :return: found_parameters : numpy.array
        List of parameters values after optimisation routine.
    """

    if len(initial_point) != len(data_exp.fitting_instructions.fitted_params_annot):
        raise ValueError('The initial point should have the same length as the fitted parameters annotations (defined in data_exp.fitting_instructions')

    if len(boundaries_low) != len(data_exp.fitting_instructions.fitted_params_annot):
        raise ValueError('The lower boundaries should have the same length as the fitted parameters annotations (defined in data_exp.fitting_instructions')

    if len(boundaries_high) != len(data_exp.fitting_instructions.fitted_params_annot):
        raise ValueError('The higher boundaries should have the same length as the fitted parameters annotations (defined in data_exp.fitting_instructions')

    fit_values = np.concatenate(data_exp.values)

    problem = pints.SingleOutputProblem(model = MyModel(), times = np.linspace(0,1,len(fit_values)), values = fit_values)
    boundaries = pints.RectangularBoundaries(boundaries_low, boundaries_high)
    error_measure = pints.SumOfSquaresError(problem)
    found_parameters, found_value = pints.optimise(error_measure, initial_point, boundaries=boundaries, method=pints.XNES)
    print(data_exp.fitting_instructions.fitted_params_annot)
    print(found_parameters)
    return found_parameters


def MCMC_inference_model_params(starting_point, max_iter=4000, adapt_start=1000, log_prior = None, mmt_model_filename = None,
                                chain_filename = None, pdf_filename = None, log_likelihood=None, method = None):
    """
    Runs a MCMC routine for the selected model

    :param starting_point: list
        List of starting values for the MCMC for the optimisation parameters. Must have the same length as
        data_exp.fitting_parms_annot + 1 (for Noise). len(starting_point) defines the amount of MCMC chains.
    :param log_prior: pints.log_prior()
        Type of prior. If not specified, pints.UniformLogPrior
    :param mmt_model_filename: str
        location of the mmt model to run if different from the one loaded previously. It will replace the
        sabs_pkpd.constants.s myokit.Simulation() already present
    :param log_likelihood: pints.LogLikelihood
        Type of log likelihood. If not specified, pints.UnknownNoiseLogLikelihood
    :param method: pints.method
        method of optimisation. If not specified, pints.AdaptiveCovarianceMCMC.
    :return: chains
        The chain for the MCMC routine.

    """

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
        log_prior = pints.UniformLogPrior(np.array(starting_point[0] * 0.5).tolist(), np.array(starting_point[0] * 2).tolist())

    fit_values = np.concatenate(sabs_pkpd.constants.data_exp.values)

    problem = pints.SingleOutputProblem(model, times=np.linspace(0,1,len(fit_values)), values=fit_values)

    # Create a log-likelihood function if not specified (adds an extra parameter!)
    if log_likelihood is not None:
        pass
    else:
        log_likelihood = pints.UnknownNoiseLogLikelihood(problem)

    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    if method is not None:
        pass
    else:
        method = pints.AdaptiveCovarianceMCMC

    # Create mcmc routine
    mcmc = pints.MCMCSampling(log_posterior, 1, starting_point, method=method)

    # Add stopping criterion
    mcmc.set_max_iterations(max_iter)
    # Start adapting after adapt_start iterations
    mcmc.set_initial_phase_iterations(adapt_start)

    if chain_filename is not None:
        mcmc.set_chain_filename(chain_filename) # 'C:/Users/yanral/Documents/Software Development/Examples/Example fitting protocol/MCMC_chain_log.csv'
    if pdf_filename is not None:
        mcmc.set_log_pdf_filename(pdf_filename) # 'C:/Users/yanral/Documents/Software Development/Examples/Example fitting protocol/MCMC_log_pdf_log.csv'

    # Run!
    print('Running...')
    chains = mcmc.run()
    print('Done!')

    return chains

def plot_distribution_map(mcmc_chains, expected_value=None, chain_index=0, fig_size=(15,15), explor_iter = 1000):
    """
    Plots a figure with histograms of distribution of all parameters used for MCMC, as well as 2D distributions of each
    couple of parameters to eventually identify linear relationships
    :param mcmc_chains: list
    List of length number of chains, each chain having a size (number of iterations, number of parameters + 1 for Noise)
    :param expected_value: list
    List of length number of parameters used for MCMC routine. If specified, it adds green lines for the expected
    value of each parameter
    :param chain_index: int
    Index of the chain for which the parameters distribution map has to be plotted. If not specified, the first chain is
    considered
    :param fig_size: tuple
    Defines the size of the figure for the plot
    :return: None
    """
    if chain_index > len(mcmc_chains)-1:
        raise ValueError('This MCMC output does not have enough chains to reach for chain no. ' + chain_index + '. Only ' +
                         len(mcmc_chains) + ' chains in this MCMC output.')

    n_param = sabs_pkpd.constants.n

    if fig_size is None:
        fig_size = (15, 15)

    start_parameter = mcmc_chains[0][0, :]
    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)

    for i in range(n_param):
        for j in range(n_param):

            # Create subplot
            if i == j:
                # Plot the diagonal
                hist_1d(mcmc_chains[chain_index][explor_iter:, i], ax=axes[i, j])
                if expected_value is not None:
                    axes[i, j].axvline(expected_value[i], c='g')
                axes[i, j].axvline(start_parameter[i], c='b')
                axes[i, j].legend()
            elif i < j:
                # Upper triangle: No plot
                axes[i, j].axis('off')
            else:
                # Lower triangle: Pairwise plot
                plot_kde_2d(j, i, mcmc_chains, ax=axes[i, j], chain_index=chain_index)
                if expected_value is not None:
                    axes[i, j].axhline(expected_value[i], c='g')
                    axes[i, j].axvline(expected_value[j], c='g')
                axes[i, j].axhline(start_parameter[i], c='b')
                axes[i, j].axvline(start_parameter[j], c='b')

            # Adjust the tick labels
            if i < n_param - 1:
                # Only show x tick labels for the last row
                axes[i, j].set_xticklabels([])
            else:
                # Rotation the x tick labels to fit in the plot
                for tl in axes[i, j].get_xticklabels():
                    tl.set_rotation(45)
            if j > 0:
                # Only show y tick labels for the first column
                axes[i, j].set_yticklabels([])

        # Add labels to the subplots at the edges
        axes[i, 0].set_ylabel('parameter %d' % (i + 1))
        axes[-1, i].set_xlabel('parameter %d' % (i + 1))

    plt.show()

    return None


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
    kernel = stats.gaussian_kde(x)
    f = kernel(x1)
    hist = ax.hist(x, bins=x2, density=True)
    ax.plot(x1, f)

    return None


def plot_kde_2d(i, j, mcmc_chains, ax, chain_index=0):
    """
    Returns the 2D distribution of parameter j versus parameter i
    :param i: int
    Index of the first parameter for the distriubtion 2D map
    :param j: int
    Index of the second parameter for the distribution 2D map
    :param mcmc_chains: list
    List of length number of chains, each chain having a size (number of iterations, number of parameters + 1 for Noise)
    :param ax :matplotlib.axes._subplots.AxesSubplot
    Axes of the figure that we want to plot the histogram on
    :param chain_index: int
    Index of the chain for which the parameters distribution map has to be plotted. If not specified, the first chain is
    considered
    :return: None
    """
    ax.set_xlabel('parameter ' + str(i))
    ax.set_ylabel('parameter '+ str(j))
    x = mcmc_chains[chain_index][:, i]
    y = mcmc_chains[chain_index][:, j]
    plt.show()

    # Get minimum and maximum values
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # Plot values
    values = np.vstack([x, y])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.imshow(np.rot90(values), cmap=plt.cm.Blues, extent=[xmin, xmax, ymin, ymax])

    # Create grid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Get kernel density estimate and plot contours
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    ax.contourf(xx, yy, f, cmap='Blues')
    ax.contour(xx, yy, f, colors='k')

    # Fix aspect ratio, see: https://stackoverflow.com/questions/7965743
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])))

    return None


"""
def infer_protocol_params(protocol_optimisation_instructions):

    sabs_pkpd.constants.s = []

    for i in range(len(protocol_optimisation_instructions.list_of_models)):
        sabs_pkpd.constants.models_list.append(protocol_optimisation_instructions.list_of_models[i])
        model = sabs_pkpd.clamp_experiment.clamp_experiment_model(protocol_optimisation_instructions.list_of_models[i],
                                                                  protocol_optimisation_instructions.protocol_clamped_variable_annotation,
                                                                  protocol_optimisation_instructions.protocol_pace_annotation)
        sabs_pkpd.constants.s.append(model)

    fit_values = [9999999999]
    problem = pints.SingleOutputProblem(model=MyProtocol(), times=np.linspace(0, 1, len(fit_values)), values=fit_values)
    boundaries = pints.RectangularBoundaries(protocol_optimisation_instructions.boundaries_low, protocol_optimisation_instructions.boundaries_high)
    error_measure = pints.SumOfSquaresError(problem)
    found_parameters, found_value = pints.optimise(error_measure, initial_point, boundaries=boundaries,
                                                   method=pints.XNES)
"""

def objective(starting_times, duration, amplitude, baseline):
    time_series = sabs_pkpd.protocols.TimeSeriesFromSteps(starting_times, duration, amplitude, baseline=baseline)
    prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(time_series)

    sample_timepoints = 10000

    response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
    for i in range(len(sabs_pkpd.constants.s)):
        simulation = myokit.Simulation(model=sabs_pkpd.constants.s[i], protocol=prot)
        simulated = simulation.run(sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time,
                                   log_times=np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time))
        response[i, :] = simulation[sabs_pkpd.constants.protocol_optimisation_instructions.model_readout]

    score = 0
    for i in range(len(sabs_pkpd.constants.s)):
        score_model = 0
        for j in range(len(sabs_pkpd.constants.s) - i):
            score_model += np.sum(np.square(response[i, :] - response[j, :]))
        score_model = np.log(score_model)
        score += score_model

    return score

if __name__ == 'main':
    list_of_models = ['C:/Users/yanral/Documents/Software Development/mmt_models/lei-2019-ikr.mmt',
                      'C:/Users/yanral/Documents/Software Development/mmt_models/beattie-2018-ikr.mmt']
    clamped_variable_model_annotation = 'membrane.v'
    pacing_model_annotation = 'engine.pace'
    simulation_time = 1500
    readout = 'ikr.IKr'
    sabs_pkpd.constants.protocol_optimisation_instructions = sabs_pkpd.constants.Protocol_optimisation_instructions(list_of_models, clamped_variable_model_annotation, pacing_model_annotation, simulation_time, readout)

    # Pre load all the models required for protocol optimisation
    sabs_pkpd.constants.s = []

    for i in range(len(sabs_pkpd.constants.protocol_optimisation_instructions.models)):
        model = sabs_pkpd.clamp_experiment.clamp_experiment_model(sabs_pkpd.constants.protocol_optimisation_instructions.models[i],
                                                                  sabs_pkpd.constants.protocol_optimisation_instructions.clamped_variable_model_annotation,
                                                                  sabs_pkpd.constants.protocol_optimisation_instructions.pacing_model_annotation)
        sabs_pkpd.constants.s.append(model)

    # Define the starting point for the optimisation
    starting_times = [100, 500, 1000, 2000, 3000, 3200, 3400, 3500, 3800]
    duration = [200, 200, 400, 800, 200, 500, 250, 150, 500, 100]
    amplitude = [30, 40, 60, 50, 25, 30, 40, 15, 25, 10]
    baseline = -85
    protocol = np.concatenate((starting_times, duration, amplitude, baseline))

    """
    if len(parameters) % 3 != 1:
        raise ValueError('3 parameters should be provided by step, plus one for baseline')

    nb_steps = len(parameters) // 3
    starting_times = parameters[0:nb_steps]
    durations = parameters[nb_steps:2 * nb_steps]
    amplitudes = parameters[2 * nb_steps:-1]
    baseline = parameters[-1]
    """
    np_objective = lambda protocol_paramaters : objective(protocol_paramaters[0 : len(protocol_paramaters)//3],
                                                          protocol_parameters[len(protocol_paramaters)//3 : 2*len(protocol_paramaters)//3],
                                                          protocol_parameters[2*len(protocol_paramaters)//3 : 3*len(protocol_paramaters)//3],
                                                          protocol_parameters[-1])

    res = scipy.optimize.maximize(np_objective, x0=protocol, method='L-BFGS-B',
                                  options={'eps': 5e-2, 'disp': True, 'maxiter': 1000})

