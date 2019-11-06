import numpy as np

def load_data_file(filename, headers = True : bool):

    # Data should be provided in 4 columns : Time in hours, Concentration in ng/mL, Dose in mg/kg, Experiment number
    data = np.loadtxt(filename, delimiter = ',', skiprows = int(headers))
    if type(data[0][0]) == str :
        raise ValueError('The CSV file is not in the standard format. Please refer to the documentation. (More than one line of headers)')
    if len(data[0])>4:
        print('The provided CSV file has more than 4 columns. Proceeding anyway...')
    data_times = [list(np.array(data)[0:5,0]), list(np.array(data)[5:10,0])]
    values = [list(np.array(data)[0:5,1]), list(np.array(data)[5:10,1])]