import sabs_pkpd
import io
import pytest
import numpy as np
import matplotlib.pyplot as plt
import pints

def test_infer_params():

    # Fix the variables which have to be global
    sabs_pkpd.constants.n = 2
    sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt('./tests/test resources/pints_problem_def_test.mmt')

    # Define all the conditions for parameters inference
    initial_point = [0.5, 0.5]
    boundaries_low = [0, 0]
    boundaries_high = [1, 1]

    sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.load_data_file('./tests/test resources/load_data_test.csv')
    sabs_pkpd.constants.data_exp.Add_fitting_instructions(['constants.unknown_cst', 'constants.unknown_cst2'], 'constants.T', 'comp1.y')

    # Fix the random seed so that the parameter inference always returns the same fitted parameters
    np.random.seed(19580)

    inferred_params = sabs_pkpd.pints_problem_def.infer_params(initial_point, sabs_pkpd.constants.data_exp, boundaries_low, boundaries_high)
    diff = inferred_params - np.array([0.1, 0.1])
    print(inferred_params)
    assert np.linalg.norm(diff) < 0.01

def test_MCMC_inference_model_params():
    # Prepare the data for example problem
    filename = 'C:/Users/yanral/Documents/Software Development/Examples/Example loading CellML model/tentusscher_2006_pints_and_Chons_hERG.mmt'

    fitting_param_annot = ['ikr.scale_kr', 'ical.scale_cal']
    sabs_pkpd.constants.n = 2

    sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(filename)

    times = [np.linspace(0,1000,1001)]
    val = sabs_pkpd.constants.s.run(1001, log_times =times[0])
    values = [val['membrane.V']]
    exp_nums_list = [1]
    exp_conds_list = [310]
    exp_cond_annot = 'phys.T'
    readout = 'membrane.V'
    sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.Data_exp(times,values, exp_nums_list, exp_conds_list)
    sabs_pkpd.constants.data_exp.Add_fitting_instructions(fitting_param_annot,exp_cond_annot, readout)

    RealValue = [1, 1]
    Noise_sigma=0.5 # variance of N distributed noise to add to the trace

    set_to_test = np.zeros(len(RealValue))
    for i in range(len(RealValue)):
      set_to_test[i] = np.random.uniform(low=0.9, high=1.11, size=None)*RealValue[i]
    starting_point = [np.array(list(set_to_test) + [Noise_sigma])]

    chains = sabs_pkpd.pints_problem_def.MCMC_inference_model_params(starting_point,max_iter=3000)
