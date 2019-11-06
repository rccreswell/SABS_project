import pints
import myokit
import numpy as np


class MyModel(pints.ForwardModel):
    def n_parameters(self):
        # Define the amount of fitted parameters
        return n_params
