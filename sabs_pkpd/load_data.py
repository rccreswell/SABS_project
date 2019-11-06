import numpy as np

class Data_exp():

    def __init__ (self, times, values, experiment_number, experiment_condition):
        self.times = []
        self.values = []
        self.experiment_number = []
        self.experiment_condition = []


    def add_times(self, data_in):
        self.values = data_in

    def add_values(self, data_in):
        self.values = data_in

    def add_exp_nums(self, data_in):
        self.values = data_in

    def add_exp_conds(self, data_in):
        self.values = data_in


def load_data_file(Data_class: Data_exp, filename, headers: bool = True):

    # Data should be provided in 4 columns : time, data, experiment number, experiment condition,
    data = np.loadtxt(filename, delimiter = ',', skiprows = int(headers))

    if type(data[0][0]) == str :
        raise ValueError('The CSV file is not in the standard format. Please refer to the documentation. (More than one line of headers)')
    if len(data[0])>4:
        print('The provided CSV file has more than 4 columns. Proceeding anyway...')

    Data_class.add_times(data[:, 0])
    Data_class.add_values(data[:, 1])
    Data_class.add_exp_nums(data[:, 2])
    Data_class.add_exp_conds(data[:, 3])