import numpy as np
import sabs_pkpd
import scipy

list_of_models = ['C:/Users/yanral/Documents/Software Development/mmt_models/lei-2019-ikr.mmt',
                  'C:/Users/yanral/Documents/Software Development/mmt_models/beattie-2018-ikr.mmt']
clamped_variable_model_annotation = 'membrane.V'
pacing_model_annotation = ['engine.pace', 'engine.pace']
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
        sabs_pkpd.constants.protocol_optimisation_instructions.pacing_model_annotation[i])
    sabs_pkpd.constants.s.append(model)

# Define the starting point for the optimisation
events_list = [[0, 100, 20, 550, 65, 22, 450, 50, 400, 50], [-85, -50, -55, 20, -30, 20, 0, -40, 10, -75]]


np_objective = lambda events: sabs_pkpd.optimize_protocol_model_distinction.objective(
    events[:int(len(events)/2)], events[int(len(events)/2):])

# The protocol will be passed to the model via the objective function which is a lambda function, i.e. taking 1D array
# as input
x0=np.reshape(events_list, (np.shape(events_list)[1] * 2, 1))

constraint_matrix = np.transpose(np.zeros(np.shape(x0)))
constraint_matrix[ : int(len(x0)/2)] = 1.0
constraints_dict = scipy.optimize.LinearConstraint(A=constraint_matrix, lb=0, ub=sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time)

print('Starting score point: ' + str(np_objective(x0)) + '\n')
print('Starting optimisation...')
res = scipy.optimize.minimize(np_objective, x0=x0, method='SLSQP', constraints=constraints_dict,
                              options={'eps': 0.01, 'disp': True, 'maxiter': 10})
