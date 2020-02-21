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
        out = sabs_pkpd.run_model.simulate_data(parameters, sabs_pkpd.constants.s, sabs_pkpd.constants.data_exp, pre_run = sabs_pkpd.constants.pre_run)
        out = np.concatenate(out)
        return out


def infer_params(initial_point, data_exp, boundaries_low, boundaries_high, pints_method = pints.XNES):
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

    sabs_pkpd.constants.n = len(sabs_pkpd.constants.data_exp.fitting_instructions.fitted_params_annot)

    problem = pints.SingleOutputProblem(model = MyModel(), times = np.linspace(0,1,len(fit_values)), values = fit_values)
    boundaries = pints.RectangularBoundaries(boundaries_low, boundaries_high)
    error_measure = pints.SumOfSquaresError(problem)
    found_parameters, found_value = pints.optimise(error_measure, initial_point, boundaries=boundaries, method=pints_method)
    print(data_exp.fitting_instructions.fitted_params_annot)
    print(found_parameters)
    return found_parameters, found_value


def MCMC_inference_model_params(starting_point, max_iter=4000, adapt_start=1000, log_prior=None,
                                mmt_model_filename=None, chain_filename = None, pdf_filename=None,
                                log_likelihood='UnknownNoiseLogLikelihood', method='HaarioBardenetACMC', sigma0=None):
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
        log_prior = pints.UniformLogPrior(np.array(mini * 0.5).tolist() * len(starting_point),
                                          np.array(maxi * 2).tolist() * len(starting_point))

    fit_values = np.concatenate(sabs_pkpd.constants.data_exp.values)

    problem = pints.SingleOutputProblem(model, times=np.linspace(0, 1, len(fit_values)), values=fit_values)

    # Create a log-likelihood function (adds an extra parameter!)
    log_likelihood = eval('pints.'+ log_likelihood + '(problem)')

    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    method = eval('pints.' + method)

    # Create mcmc routine
    mcmc = pints.MCMCSampling(log_posterior, len(starting_point), starting_point, method=method, sigma0=sigma0)

    # Add stopping criterion
    mcmc.set_max_iterations(max_iter)
    # Start adapting after adapt_start iterations
    mcmc.set_initial_phase_iterations(adapt_start)
    if adapt_start > max_iter:
        raise ValueError('The maximum number of iterations should be higher than the adapting phase length. Got ' +
                         str(max_iter) + ' maximum iterations, ' + str(adapt_start) + ' iterations in adapting phase')

    if chain_filename is not None:
        mcmc.set_chain_filename(chain_filename)
    if pdf_filename is not None:
        mcmc.set_log_pdf_filename(pdf_filename)

    # Run!
    print('Running...')
    chains = mcmc.run()
    print('Done!')

    return chains


def plot_distribution_parameters(mcmc_chains, bound_min, bound_max, chain_index=0, fig_size=(15,15), explor_iter=1000):
    if chain_index > len(mcmc_chains)-1:
        raise ValueError('This MCMC output does not have enough chains to reach for chain no. ' + str(chain_index) +
                         '. Only ' + str(len(mcmc_chains)) + ' chains in this MCMC output.')
    n_columns = 4
    n_rows = 1 + len(mcmc_chains[0,0])//4

    fig, axes = plt.subplots(n_rows, n_columns, figsize=fig_size)

    for i in range(len(mcmc_chains[0,0])-1):
        ax = axes[i//4, i%4]
        hist_1d(mcmc_chains[chain_index][explor_iter:, i], ax=ax)
        ax.set_title(sabs_pkpd.constants.data_exp.fitting_instructions.fitted_params_annot[i])
        ax.set_xlim((bound_min[i], bound_max[i]))

    plt.show()

    return fig, axes


def plot_distribution_map(mcmc_chains, expected_value=None, chain_index=0, fig_size=(15,15), explor_iter = 1000,
                          bound_max = None, bound_min = None):
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
        raise ValueError('This MCMC output does not have enough chains to reach for chain no. ' + str(chain_index) +
                         '. Only ' + str(len(mcmc_chains)) + ' chains in this MCMC output.')

    sabs_pkpd.constants.n = len(mcmc_chains[0][0, :])-1
    n_param = sabs_pkpd.constants.n

    start_parameter = mcmc_chains[0][0, :]
    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)

    for i in range(n_param):
        for j in range(n_param):

            # Create subplot
            if i == j:
                # Plot the diagonal
                if expected_value is not None:
                    axes[i, j].axvline(expected_value[i], c='g')
                if bound_max is not None:
                    axes[i, j].axvline(bound_max[i], c='r')
                if bound_min is not None:
                    axes[i, j].axvline(bound_min[i], c='r')
                axes[i, j].axvline(start_parameter[i], c='b')
                hist_1d(mcmc_chains[chain_index][:explor_iter, i], ax=axes[i, j])

            elif i < j:
                # Upper triangle: No plot
                axes[i, j].axis('off')
            else:
                # Lower triangle: Pairwise plot
                plot_kde_2d(j, i, mcmc_chains[:, :explor_iter, :], ax=axes[i, j], chain_index=chain_index)
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

    hist = ax.hist(x, bins=x2, density=True)
    kernel = stats.gaussian_kde(x)
    f = kernel(x1)
    ax.plot(x1, f)

    return None


def plot_kde_2d(i, j, mcmc_chains, ax, chain_index):
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

    return fig, axes
