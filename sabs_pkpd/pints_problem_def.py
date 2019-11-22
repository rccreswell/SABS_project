import pints
import myokit
import numpy as np
import sabs_pkpd


class MyModel(pints.ForwardModel):
    def n_parameters(self):
        # Define the amount of fitted parameters
        return sabs_pkpd.constants.n

    def simulate(self, parameters, times):

        out = sabs_pkpd.run_model.simulate_data(parameters, sabs_pkpd.constants.s, sabs_pkpd.constants.data_exp, pre_run = 0)
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
    return found_parameters

def MCMC_routine():
    # Then create an instance of our new model class
    model = Model()
    # In[SimulateTrace]:
    values = model.simulate(RealValue, times)  # Run the Simulation
    # Add noise
    valuesWithNoise = np.random.normal(0, Noise_sigma, len(values)) + values

    plt.figure()

    plt.plot(np.linspace(0, 3999, 4000), valuesWithNoise, label='values to fit')
    plt.plot(np.linspace(0, 3999, 4000), values, label='Org Signal')
    # plt.plot(np.linspace(0, 3999, 4000), model.simulate(chains[0][4999],times), label = 'end point')
    plt.legend()

    # In[]:
    times = np.linspace(0, 1000, np.size(values))
    # Starting point use optimised result
    found_parameters_noise = [np.array(list(set_to_test) + [Noise_sigma])]
    xs = found_parameters_noise
    log_prior = pints.UniformLogPrior(np.array(xs[0] * 0.5).tolist(), np.array(xs[0] * 2).tolist())
    # log_prior = pints.UniformLogPrior([6, 0.25, 0.8, 1, 1, 0.8, 3, 0.8, 0.9, 1.5, 1.5, 2.5, 0.25, 0],[10, 0.8, 2.2, 2.5, 4, 2.5, 5.5, 5, 5, 4.5, 4.5, 4.5, 0.6, 50])

    problem = pints.SingleOutputProblem(model, times, valuesWithNoise)

    # Create a log-likelihood function (adds an extra parameter!)
    log_likelihood = pints.UnknownNoiseLogLikelihood(problem)
    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    # Choose starting points for 3 mcmc chains
    # xs = found_parameters_noise

    # log_prior = pints.UniformLogPrior([np.array(xs[0]*0.5).tolist()],[np.array(xs[0]*2).tolist()])
    # Create mcmc routine
    mcmc = pints.MCMCSampling(log_posterior, 1, xs, method=pints.AdaptiveCovarianceMCMC)

    # Add stopping criterion
    mcmc.set_max_iterations(5000)
    initial_point = 1000
    # Start adapting after 1000 iterations
    mcmc.set_initial_phase_iterations(initial_point)
    mcmc.set_chain_filename('C:/Users/wangk39/Desktop/MCMC/Chain_Verapamil_9__simulatedNormalNoise_CloseStart2.csv')
    mcmc.set_log_pdf_filename('C:/Users/wangk39/Desktop/MCMC/LogPDF_Verapamil_9__simulatedNormalNoise_CloseStart2.csv')

    # Disable verbose mode
    # mcmc.set_verbose(False)

    # Run!
    print('Running...')
    chains = mcmc.run()
    print('Done!')

    # In[57]:
    import scipy.stats as stats
    real_parameters = [RealValue] + [Noise_sigma]

    # real_parameters=[9.0, 0.4, 1.1, 1.9, 1.1, 1.1, 4, 4, 2.6, 3.0, 3.8, 3.8, 0.35, 1]

    def plot_kde_1d(x, ax):
        """ Creates a 1d histogram and an estimate of the PDF using KDE. """
        xmin = np.min(x)
        xmax = np.max(x)
        x1 = np.linspace(xmin, xmax, 100)
        x2 = np.linspace(xmin, xmax, 50)
        kernel = stats.gaussian_kde(x)
        f = kernel(x1)
        hist = ax.hist(x, bins=x2, normed=True)
        ax.plot(x1, f)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Probability density')
    plot_kde_1d(chains[0][1000:, 0], ax)

    plt.show()

    # In[57]:
    def plot_kde_2d(x, y, ax):
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

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('parameter 1')
    ax.set_ylabel('parameter 2')
    plot_kde_2d(chains[0][:, 0], chains[0][:, 1], ax)
    plt.show()

    # In[58]:
    # real_parameters= list(set_to_test) + [3]
    n_param = log_likelihood.n_parameters()
    fig_size = (30, 30)
    # real_parameters=[9.5, 0.45, 1.3, 1.8, 1.4, 1.5, 4.5, 3.4, 1.9, 2.4, 3.8, 2.9, 0.6,1]
    # real_parameters=[9.0, 0.4, 1.1, 1.9, 1.1, 1.1, 4, 4, 2.6, 3.0, 3.8, 3.8, 0.35, 1]
    start_parameter = chains[0][0, :]
    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)
    for i in range(n_param):
        for j in range(n_param):

            # Create subplot
            if i == j:
                # Plot the diagonal
                plot_kde_1d(chains[0][:, i], ax=axes[i, j])
                axes[i, j].axvline(real_parameters[i], c='g')
                axes[i, j].axvline(start_parameter[i], c='b')
                axes[i, j].legend()
            elif i < j:
                # Upper triangle: No plot
                axes[i, j].axis('off')
            else:
                # Lower triangle: Pairwise plot
                plot_kde_2d(chains[0][:, j], chains[0][:, i], ax=axes[i, j])
                axes[i, j].axhline(real_parameters[i], c='g')
                axes[i, j].axvline(real_parameters[j], c='g')

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
    # In[]:
    plt.show()
    fig_size = (60, 60)
    # .figure()
    plt.plot(np.linspace(0, 3999, 4000), values[0], label='values to fit')
    plt.plot(np.linspace(0, 3999, 4000), model.simulate(set_to_test, times), label='starting point')
    plt.plot(np.linspace(0, 3999, 4000), model.simulate(chains[0][4999, 0:13], times), label='EndChain')
    # plt.plot(np.linspace(0, 3999, 4000), model.simulate([9.5, 0.45, 1.3, 1.8, 1.4, 1.5, 4.5, 3.4, 1.9, 2.4, 3.8, 2.9, 0.6],times), label = 'RealValue')
    plt.plot(np.linspace(0, 3999, 4000), model.simulate(RealValue, times), label='RealValue')
    # plt.plot(np.linspace(0, 3999, 4000), model.simulate(chains[0][4999],times), label = 'end point')
    plt.legend()