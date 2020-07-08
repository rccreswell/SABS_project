# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:28:36 2019

@author: yanral
"""

#%% Import libraries
import myokit
import pints
import matplotlib.pyplot as plt
import numpy as np

model, prot, script = myokit.load('./PKPD_linear.mmt')


#%% Read data

# Data is provided in 3 columns : Time in hours, Concentration in ng/mL,
# Dose in mg/kg
data = np.loadtxt('./PK_Excercise4.csv', delimiter=',', skiprows=1)
data_times = [list(np.array(data)[0:5, 0]), list(np.array(data)[5:10, 0])]
values = [list(np.array(data)[0:5, 1]), list(np.array(data)[5:10, 1])]


#%% Dose computation
mouse_weight = 25  # in g
IV_dose = [data[0][2], data[5][2]]  # in mg/kg
IV_deliv_time = 0.01  # in hours. Make sure it matches with the protocol of
# .mmt model

# Compute and convert dose(t) in ng
total_drug = [IV_dose[i] * 1e6 * mouse_weight * 1e-3
              for i in range(len(IV_dose))]
dose = [total_drug[i] / IV_deliv_time for i in range(len(IV_dose))]


#%% Use PINTS parameter inference
class MyModel(pints.ForwardModel):
    def n_parameters(self):
        # Define the amount of fitted parameters
        return 6

    def simulate(self, parameters, times):

        s.reset()

        # Set parameters for simulation
        # Parameters : [CL, Vc, Qp1, Vp1, Qp2, Vp2]
        s.set_constant('constants.CL', parameters[0])
        s.set_constant('plasma.Vc', parameters[1])
        s.set_constant('constants.kp1', parameters[2])
        s.set_constant('compartment_1.V1', parameters[3])
        s.set_constant('constants.kp2', parameters[4])
        s.set_constant('compartment_2.V2', parameters[5])

        output = []

        # Run the model solving for all of the doses
        for i in range(len(dose)):
            # reset timer
            s.set_time(0)

            # set the right dose
            s.set_constant('administration.dose_amount', dose[i])

            # Run the simulation with starting parameters
            a = s.run(data_times[i][-1] + 0.1, log_times=data_times[i][0:5])
            out = list(a['plasma.y_c'])
            output = output + out
        # Convert output in concentration
        output = np.multiply(output, 1 / parameters[1])
        return output

#%%
#Fitting the model


#Reload the model
model, prot, script = myokit.load('./PKPD_linear.mmt')

s = myokit.Simulation(model, prot)

# Format the values list to use it in PINTS
fit_values = values[0] + values[1]

# Calling PINTS library
# Parameters : [CL, Vc, Qp1, Vp1, Qp2, Vp2]
initial_point = [5.4, 8.5, 18.6, 8.3, 1.7, 32.8]
problem = pints.SingleOutputProblem(model=MyModel(),
                                    times=np.linspace(0, 24, 10),
                                    values=fit_values)
boundaries = pints.RectangularBoundaries([3, 5, 7, 5, 0.5, 30],
                                         [15, 15, 30, 20, 5, 60])
error_measure = pints.SumOfSquaresError(problem)
found_parameters, found_value = pints.optimise(error_measure,
                                               initial_point,
                                               boundaries=boundaries,
                                               method=pints.XNES)


#%% Running the simulation with found parameters
# Reset the variables to initial state for the plot
s.reset()

# Use parameters returned from optimisation or user-defined parameters set
parameters = [6.7, 10.1, 17.8, 9.4, 1.4, 18.3]
plot_parameters = found_parameters
s.set_constant('constants.CL', plot_parameters[0])
s.set_constant('plasma.Vc', plot_parameters[1])
s.set_constant('constants.kp1', plot_parameters[2])
s.set_constant('compartment_1.V1', plot_parameters[3])
s.set_constant('constants.kp2', plot_parameters[4])
s.set_constant('compartment_2.V2', plot_parameters[5])
out = []

# Solve the problem with the desired parameters
for i in range(len(dose)):
    # reset timer
    s.set_time(0)

    # set the right dose
    s.set_constant('administration.dose_amount', dose[i])

    # Run the simulation with starting parameters
    # We have to add an extra value at the end of log_times to make the whole
    # times array is used. The times values have to be increasing, hence the
    # +0.1
    a = s.run(data_times[i][-1] + 0.1, log_times=data_times[i])

    # Plot the comparison between fitted model and real data for each dose
    plt.figure()
    plt.title('Fitted model vs Data')
    plt.xlabel('Time (in h)')
    plt.ylabel('Drug concentration (in ng/mL)')
    out = out + list(a['plasma.y_c'])
    out = np.multiply(out, 1 / found_parameters[1])
    plt.plot(data_times[i], out, label='Fitted model')
    plt.plot(data_times[i], values[i], label='Data')
    plt.legend()
