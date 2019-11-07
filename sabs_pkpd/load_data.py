import numpy as np
from operator import itemgetter
import myokit

class Data_exp():

    def __init__ (self,times, values, exp_nums, exp_conds):
        self.times = times
        self.values = values
        self.exp_nums = exp_nums
        self.exp_conds = exp_conds


    def add_times(self, times_in):
        self.times = times_in

    def add_values(self, data_in):
        self.values = data_in

    def add_exp_nums(self, exps_in):
        self.exp_nums= exps_in

    def add_exp_conds(self, conds_in):
        self.exp_conds = conds_in



def load_data_file(filename, headers: bool = True):

    # Data should be provided in 4 columns : time, data, experiment number, experiment condition,
    data = np.loadtxt(filename, delimiter= ',', skiprows= int(headers))

    if type(data[0][0]) == str :
        raise ValueError('The CSV file is not in the standard format. Please refer to the documentation. (More than one line of headers)')
    if len(data[0]) > 4:
        raise ValueError('The CSV file is not in the standard format. Please refer to the documentation. (Too many columns)')

    # Sorting the list in increasing times and experiment number

    data = sorted(data, key=itemgetter(2, 0))
    data = np.concatenate([i for i in data])
    data = data.reshape(len(data)//4, 4)
    exp_nums_list = list(set(data[:,2]))
    exp_conds_list = list(set(data[:, 3]))
    times = []
    values = []

    for i in range(len(exp_nums_list)):
        temp = data[ data[:, 2] == exp_nums_list[i] ]
        times.append(temp[:,0])
        values.append(temp[:,1])

    return Data_exp(times,values, exp_nums_list, exp_conds_list)
