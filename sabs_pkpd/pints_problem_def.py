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

    fit_values = np.concatenate(data_exp.values)

    problem = pints.SingleOutputProblem(model = MyModel(), times = np.linspace(0,1,len(fit_values)), values = fit_values)
    boundaries = pints.RectangularBoundaries(boundaries_low, boundaries_high)
    error_measure = pints.SumOfSquaresError(problem)
    found_parameters, found_value = pints.optimise(error_measure, initial_point, boundaries=boundaries, method=pints.XNES)
    print(data_exp.fitting_instructions.fitted_params_annot)
    return found_parameters
