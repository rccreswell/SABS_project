import sabs_pkpd

n = 2

s = []

default_state = None

data_exp = sabs_pkpd.load_data.Data_exp([], [], [], [])

pre_run = 0

protocol_optimisation_instructions = []


class Protocol_optimisation_instructions:
    def __init__(self,
                 list_of_models,
                 clamped_variable_model_annotation,
                 pacing_model_annotation,
                 simulation_time,
                 readout):
        self.models = list_of_models
        self.clamped_variable_model_annotation = \
            clamped_variable_model_annotation
        self.pacing_model_annotation = pacing_model_annotation
        self.simulation_time = simulation_time
        self.model_readout = readout
