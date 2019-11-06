import numpy as np
from operator import itemgetter

class Data_exp():

    def __init__ (self):
        self.times = []
        self.values = []
        self.experiment_number = []
        self.experiment_conditions = []


    def add_times(self, times_in):
        self.times = times_in

    def add_values(self, data_in):
        self.values = data_in

    def add_exp_nums(self, exps_in):
        self.experiment_number = exps_in

    def add_exp_conds(self, conds_in):
        self.experiment_conditions = conds_in



def load_data_file(Data_class: Data_exp, filename, headers: bool = True):

    # Data should be provided in 4 columns : time, data, experiment number, experiment condition,
    data = np.loadtxt(filename, delimiter = ',', skiprows = int(headers))

    if type(data[0][0]) == str :
        raise ValueError('The CSV file is not in the standard format. Please refer to the documentation. (More than one line of headers)')
    if len(data[0]) > 4:
        raise ValueError('The CSV file is not in the standard format. Please refer to the documentation. (Too many columns)')

    #Sorting the list in increasing times and experiment number

    data = sorted(data, key=itemgetter(2, 0))
    data = np.concatenate([i for i in data])
    data = data.reshape(len(data)//4, 4)

    Data_class.add_times(data[:, 0])
    Data_class.add_values(data[:, 1])
    Data_class.add_exp_nums(data[:, 2])
    Data_class.add_exp_conds(data[:, 3])