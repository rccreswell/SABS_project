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
starting_times = [100, 500, 600, 1000, 2000, 3000, 3200, 3400, 3500, 3800]
duration = [200, 200, 400, 800, 200, 500, 250, 150, 500, 100]
amplitude = [30, 40, 60, 50, 25, 30, 40, 15, 25, 10]
baseline = -85
protocol = np.concatenate((starting_times, duration, amplitude))
protocol = np.insert(protocol, len(protocol), baseline)


np_objective = lambda protocol_parameters: sabs_pkpd.optimize_protocol_model_distinction.objective(
    protocol_parameters[0: len(protocol_parameters) // 3],
    protocol_parameters[len(protocol_parameters) // 3: 2 * len(
        protocol_parameters) // 3],
    protocol_parameters[2 * len(protocol_parameters) // 3: 3 * len(
        protocol_parameters) // 3],
    protocol_parameters[-1])

print('Starting optimisation...')

res = scipy.optimize.minimize(np_objective, x0=protocol, method='Nelder-Mead',
                              options={'fatol': 1e10, 'disp': True, 'maxiter': 10})