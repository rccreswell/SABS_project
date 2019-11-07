import pints
import myokit
import numpy as np
import sabs_pkpd

class MyModel(pints.ForwardModel):
    def n_parameters(self):
        # Define the amount of fitted parameters
        ''' I have no idea how to make the user change that (for now) '''
        return n

    def simulate(self, parameters, times):
        sabs_pkpd.run_model.simulate_data(fitted_params, parameters, exp_cond_annot, s, read_out, data, pre_run = 0)

