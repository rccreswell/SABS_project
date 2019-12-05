import numpy as np
import sabs_pkpd
import scipy

list_of_models = ['C:/Users/yanral/Documents/Software Development/mmt_models/lei-2019-ikr.mmt',
                  'C:/Users/yanral/Documents/Software Development/mmt_models/beattie-2018-ikr.mmt']
clamped_variable_model_annotation = 'membrane.V'
pacing_model_annotation = 'engine.pace'
simulation_time = 1500
readout = 'ikr.IKr'
sabs_pkpd.constants.protocol_optimisation_instructions = sabs_pkpd.constants.Protocol_optimisation_instructions(
    list_of_models, clamped_variable_model_annotation, pacing_model_annotation, simulation_time, readout)

# Pre load all the models required for protocol optimisation
sabs_pkpd.constants.s = []

for i in range(len(sabs_pkpd.constants.protocol_optimisation_instructions.models)):
    model = sabs_pkpd.clamp_experiment.clamp_experiment_model(
        sabs_pkpd.constants.protocol_optimisation_instructions.models[i],
        sabs_pkpd.constants.protocol_optimisation_instructions.clamped_variable_model_annotation,
        sabs_pkpd.constants.protocol_optimisation_instructions.pacing_model_annotation)
    sabs_pkpd.constants.s.append(model)

# Define the starting point for the optimisation
events_list = [[0, 100, 20, 550, 65, 22, 450, 50, 400, 50], [-85, -50, -55, 20, -30, 20, 0, -40, 10, -75]]

# The protocol will be passed to the model via the objective function which is a lambda function, i.e. taking 1D array
# as input
events_list = np.reshape(events_list, (np.shape(events_list)[0] * np.shape(events_list)[1], 1))

np_objective = lambda protocol_parameters: sabs_pkpd.optimize_protocol_model_distinction.objective(
    protocol_parameters[0: len(protocol_parameters) // 3],
    protocol_parameters[len(protocol_parameters) // 3: 2 * len(protocol_parameters) // 3],
    protocol_parameters[2 * len(protocol_parameters) // 3: 3 * len(protocol_parameters) // 3],
    protocol_parameters[-1])

low_bound = np.zeros(np.shape(events_list))
low_bound[2 * len(events_list) // 3: 3 * len(events_list) // 3] = - 60
low_bound[-1] = -100

up_bound = np.zeros(np.shape(events_list))
up_bound[2 * len(events_list) // 3: 3 * len(events_list) // 3] = - 60
up_bound[-1] = -100

print('Starting optimisation...')
res = scipy.optimize.minimize(np_objective, x0=events_list, method='Nelder-Mead',
                              options={'fatol': 1e10, 'disp': True, 'maxiter': 10})