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
