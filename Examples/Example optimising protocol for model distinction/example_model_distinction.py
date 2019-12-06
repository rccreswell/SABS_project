import numpy as np
import sabs_pkpd
import scipy
import myokit
import matplotlib.pyplot as plt
import time

list_of_models = ['C:/Users/yanral/Documents/Software Development/mmt_models/lei-2019-ikr.mmt',
                  'C:/Users/yanral/Documents/Software Development/mmt_models/beattie-2018-ikr.mmt',
                  'C:/Users/yanral/Documents/Software Development/mmt_models/ten-tusscher-2006-ikr.mmt']
clamped_variable_model_annotation = 'membrane.V'
pacing_model_annotation = ['engine.pace', 'engine.pace', 'engine.pace']
simulation_time = 2600
sample_timepoints = 2600
readout = 'ikr.IKr'
sabs_pkpd.constants.protocol_optimisation_instructions = sabs_pkpd.constants.Protocol_optimisation_instructions(
    list_of_models, clamped_variable_model_annotation, pacing_model_annotation, simulation_time, readout)


# Define the starting point for the optimisation
events_list = [[10, 100, 400, 100, 400, 100, 400, 100, 400, 100, 40], [-85, -60, -85, -50, -85, -40, -85, -30, -85, -20, -85]]
prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(events_list[0], events_list[1])



# Pre load all the models required for protocol optimisation
sabs_pkpd.constants.s = []
sabs_pkpd.constants.pre_run = 0

for i in range(len(sabs_pkpd.constants.protocol_optimisation_instructions.models)):
    model = sabs_pkpd.clamp_experiment.clamp_experiment_model(
        sabs_pkpd.constants.protocol_optimisation_instructions.models[i],
        sabs_pkpd.constants.protocol_optimisation_instructions.clamped_variable_model_annotation,
        sabs_pkpd.constants.protocol_optimisation_instructions.pacing_model_annotation[i])
    sabs_pkpd.constants.s.append(myokit.Simulation(model=model, protocol=prot))


# The protocol will be passed to the model via the objective function which is a lambda function, i.e. taking 1D array
# as input
x0 = np.reshape(events_list, (np.shape(events_list)[1] * 2))

# Define the boundaries for
constraint_matrix = np.zeros((1, len(events_list[0])*2))
constraint_matrix[0, : int(len(x0)/2)] = 1.0
constraints_dict = scipy.optimize.LinearConstraint(A=constraint_matrix, lb=0, ub=sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time)

low_bounds = np.zeros(np.shape(x0))
low_bounds[:int(len(x0)/2)] = 5
low_bounds[int(len(x0)/2):] = -110

up_bounds = np.zeros(np.shape(x0))
up_bounds[:int(len(x0)/2)] = 500
up_bounds[int(len(x0)/2):] = 50
boundaries = scipy.optimize.Bounds(low_bounds, up_bounds)


# Define the objective function used
np_objective = lambda events: sabs_pkpd.optimize_protocol_model_distinction.objective_step_phase(
    events[:int(len(events)/2)], events[int(len(events)/2):], sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, normalise_output=False)


print('Starting score point: ' + str(np_objective(x0)) + '\n')
t0 = time.time()
print('Starting optimisation...')
res = scipy.optimize.minimize(np_objective, x0=x0, method='SLSQP', bounds=boundaries, constraints=constraints_dict,
                              options={'disp': True, 'maxiter': 10000})
t1 = time.time()
print('Optimisation time is ' + str(t1-t0) + ' seconds.')


# Plot initial point protocol and model response
fig1 = plt.figure(1)
fig_init_prot = plt.subplot(2, 1, 1)
response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
sabs_pkpd.constants.s[0].reset()
sabs_pkpd.constants.s[0].set_protocol(prot)
simulated = sabs_pkpd.constants.s[0].run(sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time*1.0001,
                           log_times=np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints))
response[0, :] = simulated[clamped_variable_model_annotation]
plt.plot(np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints),
         response[0, :])


fig_init_resp = plt.subplot(2, 1, 2)
response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
for i in range(len(sabs_pkpd.constants.s)):
    sabs_pkpd.constants.s[i].reset()
    sabs_pkpd.constants.s[i].set_protocol(protocol=prot)
    simulated = sabs_pkpd.constants.s[i].run(sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time*1.0001,
                               log_times=np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints))
    response[i, :] = simulated[sabs_pkpd.constants.protocol_optimisation_instructions.model_readout]
    response[i, :] = (response[i, :] - np.min(response[i, :]))/ (np.max(response[i, :]) - np.min(response[i, :]))
    plt.plot(np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints),
             response[i,:], label = 'model '+str(i))
plt.legend()
plt.suptitle('Initial protocol and models response')


# Plot results
prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(res.x[: int(len(res.x)/2)], res.x[int(len(res.x)/2):])

fig4 = plt.figure(2)
fig2 = plt.subplot(2, 1, 1)
response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
sabs_pkpd.constants.s[0].set_time(0)
sabs_pkpd.constants.s[0].reset()
sabs_pkpd.constants.s[0].set_protocol(prot)
simulated = sabs_pkpd.constants.s[0].run(sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time*1.0001,
                           log_times=np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints))
response[0, :] = simulated[clamped_variable_model_annotation]
plt.plot(np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints),
         response[0, :])


fig3 = plt.subplot(2, 1, 2)
response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
for i in range(len(sabs_pkpd.constants.s)):
    sabs_pkpd.constants.s[i].set_time(0)
    sabs_pkpd.constants.s[i].reset()
    sabs_pkpd.constants.s[i].set_protocol(prot)
    simulated = sabs_pkpd.constants.s[i].run(sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time*1.0001,
                               log_times=np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints))
    response[i, :] = simulated[sabs_pkpd.constants.protocol_optimisation_instructions.model_readout]
    response[i, :] = (response[i, :] - np.min(response[i, :]))/ (np.max(response[i, :]) - np.min(response[i, :]))
    plt.plot(np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints),
             response[i,:], label = 'model '+str(i))
plt.legend()
plt.suptitle('Optimised protocol and response of the selected models')