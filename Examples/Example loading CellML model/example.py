import sabs_pkpd
import matplotlib.pyplot as plt
import numpy as np

# Enter the filenames for the CellML to load and the MMT to save
cellml = 'Examples/Example loading CellML model/grandi_pasqualini_bers_2010.cellml'
mmt = 'Examples/Example loading CellML model/grandi_pasqualini_bers_2010.mmt'

# Load the CellML file and save the converted MMT model
simulation = sabs_pkpd.load_model.load_model_from_cellml(cellml, mmt)

# Run directly using Myokit and plot the output
result = simulation.run(1000, log_interval = 1)
plt.plot(result['membrane.V'])

# Run using the package
result = sabs_pkpd.run_model.quick_simulate(simulation, 1000, 'membrane.V', time_samples = np.linspace(0, 1000, 1001))
plt.plot(result[0])

