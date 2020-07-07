import sabs_pkpd
import numpy as np
import unittest


class Test(unittest.TestCase):
    def test_infer_params(self):

        # Fix the variables which have to be global
        sabs_pkpd.constants.n = 2
        sabs_pkpd.constants.s = sabs_pkpd.load_model.\
            load_simulation_from_mmt(
                './tests/test resources/pints_problem_def_test.mmt')

        # Save the default state as defined in the mmt file
        sabs_pkpd.constants.default_state = sabs_pkpd.constants.s.\
            default_state()

        # Define all the conditions for parameters inference
        initial_point = [0.5, 0.5]
        boundaries_low = [0, 0]
        boundaries_high = [1, 1]

        sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.load_data_file(
            './tests/test resources/load_data_test.csv')
        sabs_pkpd.constants.data_exp.Add_fitting_instructions(
            ['constants.unknown_cst', 'constants.unknown_cst2'],
            'constants.T',
            'comp1.y')

        # Fix the random seed so that the parameter inference always returns
        # the same fitted parameters
        np.random.seed(19580)

        inferred_params, found_value = \
            sabs_pkpd.pints_problem_def.infer_params(
                initial_point,
                sabs_pkpd.constants.data_exp,
                boundaries_low,
                boundaries_high)
        diff = inferred_params - np.array([0.1, 0.1])
        print(inferred_params)
        assert np.linalg.norm(diff) < 0.01

        # Test exceptions
        wrong_initial_point = [0.5, 0.5, 1.0]
        with self.assertRaises(ValueError) as context:
            inferred_params, found_value = \
                sabs_pkpd.pints_problem_def.infer_params(
                    wrong_initial_point,
                    sabs_pkpd.constants.data_exp,
                    boundaries_low,
                    boundaries_high)
        assert 'should have the same length' in str(context.exception)

        wrong_boundaries_low = [0.0]
        with self.assertRaises(ValueError) as context:
            inferred_params, found_value = \
                sabs_pkpd.pints_problem_def.infer_params(
                    initial_point,
                    sabs_pkpd.constants.data_exp,
                    wrong_boundaries_low,
                    boundaries_high)
        assert 'should have the same length' in str(context.exception)

        wrong_boundaries_high = [1.0]
        with self.assertRaises(ValueError) as context:
            inferred_params, found_value = \
                sabs_pkpd.pints_problem_def.infer_params(
                    initial_point,
                    sabs_pkpd.constants.data_exp,
                    boundaries_low,
                    wrong_boundaries_high)
        assert 'should have the same length' in str(context.exception)

    def test_MCMC_routine(self):
        # Set the model annotations for the MCMC routine
        fitting_param_annot = ['ikr.scale_kr', 'ical.scale_cal']
        exp_cond_annot = 'phys.T'
        readout = 'membrane.V'

        # Load model and data
        sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(
            './tests/test resources/tentusscher_2006_pints_and_Chons_hERG.mmt')
        sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.load_data_file(
            './tests/test resources/mcmc_test_data.csv')
        sabs_pkpd.constants.data_exp.Add_fitting_instructions(
            fitting_param_annot,
            exp_cond_annot,
            readout)

        # Save the default state as defined in the mmt file
        sabs_pkpd.constants.default_state = sabs_pkpd.constants.s.\
            default_state()

        # Start from a starting point close to the values of parameters used to
        # generate the synthetic data
        RealValue = [1, 1]
        Noise_sigma = 0.5  # variance of N distributed noise to add to the
        # trace

        set_to_test = np.zeros(len(RealValue))
        for i in range(len(RealValue)):
            set_to_test[i] = \
                np.random.uniform(low=0.7, high=1.5, size=None) * RealValue[i]
        starting_point = [np.array(list(set_to_test) + [Noise_sigma])]

        chains = sabs_pkpd.pints_problem_def.MCMC_routine(starting_point,
                                                          max_iter=3000)

        mean_param0 = np.mean(chains[0][:, 0])
        mean_param1 = np.mean(chains[0][:, 1])

        # assert abs(mean_param0 - 1) < 0.1
        # assert abs(mean_param1 - 1) < 0.1

        # Test the plot functions to make sure no error is returned
        sabs_pkpd.pints_problem_def.plot_MCMC_convergence(
            chains,
            expected_values=[1, 1, 0.1],
            bound_max=[2, 2, 1],
            bound_min=[0.1, 0.1, 0],
            parameters_annotations=fitting_param_annot + ['Noise'])

        sabs_pkpd.pints_problem_def.plot_distribution_parameters(
            chains,
            [0.1, 0.1, 0],
            [2, 2, 1],
            chain_index=0,
            fig_size=(15, 15),
            explor_iter=0)

        # Test exceptions
        wrong_starting_point = [np.array([1.02702835, 1.26803399, 0.5, 1.0])]
        with self.assertRaises(ValueError) as context:
            chains = sabs_pkpd.pints_problem_def.MCMC_routine(
                wrong_starting_point,
                max_iter=10)
        assert 'must have the same length' in str(context.exception)

        # test loading from mmt
        chains = sabs_pkpd.pints_problem_def.MCMC_routine(
            starting_point,
            max_iter=10,
            mmt_model_filename='./tests/test resources'
                               '/tentusscher_2006_pints_and_Chons_hERG.mmt')

        # test adapt start
        chains = sabs_pkpd.pints_problem_def.MCMC_routine(
            starting_point, max_iter=10, adapt_start=5)

        # Test adapt start exceptions
        with self.assertRaises(ValueError) as context:
            chains = sabs_pkpd.pints_problem_def.MCMC_routine(
                starting_point, max_iter=10, adapt_start=100)
        assert 'number of iterations should be higher' in \
            str(context.exception)

        # Test chain and pdf filenames
        chains = sabs_pkpd.pints_problem_def.MCMC_routine(
            starting_point,
            max_iter=10,
            chain_filename='chain.file',
            pdf_filename='pdf.file')

        return None

    def test_plot_MCMC_convergence(self):
        # Test a chain with just 1 parameter
        test_mcmc_chain = np.array([[[0.], [0.], [0.]], [[0.], [0.], [0.]]])
        f, a = sabs_pkpd.pints_problem_def.plot_MCMC_convergence(
            test_mcmc_chain, [0.0], [1.0], [-1.0])

    def test_parameter_is_state(self):
        param_annot = 'comp1.x'
        myokit_simulation = sabs_pkpd.load_model.load_simulation_from_mmt(
            './tests/test resources/pints_problem_def_test.mmt')
        assert sabs_pkpd.pints_problem_def.parameter_is_state(
            param_annot, myokit_simulation) is True

        param_annot = 'constants.T'
        assert sabs_pkpd.pints_problem_def.parameter_is_state(
            param_annot, myokit_simulation) is False

    def test_find_index_of_state(self):
        param_annot = 'comp1.x'
        myokit_simulation = sabs_pkpd.load_model.load_simulation_from_mmt(
            './tests/test resources/pints_problem_def_test.mmt')
        assert sabs_pkpd.pints_problem_def.find_index_of_state(
            param_annot, myokit_simulation) == 1
