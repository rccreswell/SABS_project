import pints
import myokit
import numpy as np
import sabs_pkpd

class FittingInstructions():

    def __init__(self, fitted_params_annot, fitted_params_values, varying_param_annot, sim_output_param_annot):
        self.fitted_params_annot = []
        self.fitted_params_values = []
        self.varying_param_annot = []
        self.sim_output_param_annot = []


class MyModel(pints.ForwardModel):
    def n_parameters(self):
        # Define the amount of fitted parameters
        ''' I have no idea how to make the user change that (for now) '''
        return n

    def simulate(self, parameters, times):
        out = sabs_pkpd.run_model.simulate_data(fitted_params, parameters, exp_cond_annot, s, read_out, data, pre_run = 0)
        out = np.concatenate([i for i in out])
        return out
