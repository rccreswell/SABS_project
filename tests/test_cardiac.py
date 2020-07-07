import numpy as np
import sabs_pkpd
import pytest
import unittest


class Test(unittest.TestCase):
    def test_inputs(self):
        AP = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        with pytest.raises(ValueError):
            sabs_pkpd.cardiac.compute_APD(AP)

        with pytest.raises(ValueError):
            sabs_pkpd.cardiac.compute_calcium_transient_duration(AP)

        AP = np.array([0, 1, 2, 3])

        time_points = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        with pytest.raises(ValueError):
            sabs_pkpd.cardiac.compute_APD(AP, time_points)

        with pytest.raises(ValueError):
            sabs_pkpd.cardiac.compute_calcium_transient_duration(AP,
                                                                 time_points)

        return None


    def test_compute_APD(self):
        mmt = \
            './tests/test resources/tentusscher_2006_pints_and_Chons_hERG.mmt'
        sabs_pkpd.constants.s = sabs_pkpd.load_model.\
                                    load_simulation_from_mmt(mmt)

        AP = sabs_pkpd.run_model.quick_simulate(sabs_pkpd.constants.s,
                                                1000,
                                                'membrane.V',
                                                pre_run=150000,
                                                fixed_params_annot=['ical.scale_cal'],
                                                fixed_params_values=[0.5],
                                                time_samples=np.linspace(0, 1000, 1001))
        AP = AP[0]
        APD90 = sabs_pkpd.cardiac.compute_APD(AP)
        expected_APD90 = 292
        APD50 = sabs_pkpd.cardiac.compute_APD(AP, repol_percentage=50)
        expected_APD50 = 255
        assert (APD90 == expected_APD90)
        assert (APD50 == expected_APD50)

        AP = sabs_pkpd.run_model.quick_simulate(sabs_pkpd.constants.s,
                                                1000,
                                                'membrane.V',
                                                pre_run=150000,
                                                fixed_params_annot=['ical.scale_cal'],
                                                fixed_params_values=[1])
        AP = AP[0]
        APD90 = sabs_pkpd.cardiac.compute_APD(AP)
        expected_APD90 = 333
        assert (np.isclose(APD90, expected_APD90, atol=1))

        times = np.hstack((np.linspace(0, 250, 1001), np.linspace(251, 1000, 250)))
        AP = sabs_pkpd.run_model.quick_simulate(sabs_pkpd.constants.s,
                                                1000,
                                                'membrane.V',
                                                pre_run=150000,
                                                time_samples=times)
        AP = AP[0]
        APD90 = sabs_pkpd.cardiac.compute_APD(AP, time_points=times)
        expected_APD90 = 329
        assert (np.isclose(APD90, expected_APD90, atol=2))

        APD90 = sabs_pkpd.cardiac.compute_APD(AP, time_points=times, upstroke_time=50)
        expected_APD90 = 330
        assert (np.isclose(APD90, expected_APD90, atol=2))

        # Test exceptions
        with self.assertRaises(ValueError) as context:
            APD = sabs_pkpd.cardiac.compute_APD(np.zeros((10, 10, 10)))
        assert 'either as a 1-D list' in str(context.exception)

        with self.assertRaises(ValueError) as context:
            APD = sabs_pkpd.cardiac.compute_APD('not_a_list')
        assert 'either as a 1-D list' in str(context.exception)

        with self.assertRaises(ValueError) as context:
            APD = sabs_pkpd.cardiac.compute_APD(AP, time_points='not_times')
        assert 'time_points should be provided ' in str(context.exception)


        # Test submitting a list of times
        APD90 = sabs_pkpd.cardiac.compute_APD(AP, time_points=list(times))
        expected_APD90 = 329
        assert (np.isclose(APD90, expected_APD90, atol=2))


        # Test upstroke time not in time points
        APD90 = sabs_pkpd.cardiac.compute_APD(AP, time_points=times, upstroke_time=50.001)
        expected_APD90 = 330
        assert (np.isclose(APD90, expected_APD90, atol=2))

        # Test abnormal AP
        AP = np.array([-80] * 10 + [100] + [-200])
        APD = sabs_pkpd.cardiac.compute_APD(AP)

        return None

    def test_compute_calcium_transient_duration(self):
        mmt = './tests/test resources/tentusscher_2006_pints_and_Chons_hERG.mmt'
        sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(mmt)

        Cai = sabs_pkpd.run_model.quick_simulate(sabs_pkpd.constants.s,
                                                 1000,
                                                 'calcium.Ca_i',
                                                 pre_run=150000,
                                                 fixed_params_annot=['ical.scale_cal'],
                                                 fixed_params_values=[0.5],
                                                 time_samples=np.linspace(0, 1000, 1001))
        Cai = Cai[0]
        CaiD90 = sabs_pkpd.cardiac.compute_calcium_transient_duration(Cai)
        expected_CaiD90 = 418
        CaiD50 = sabs_pkpd.cardiac.compute_calcium_transient_duration(Cai, repol_percentage=50)
        expected_CaiD50 = 168
        assert (np.isclose(CaiD90, expected_CaiD90, atol=1))
        assert (np.isclose(CaiD50, expected_CaiD50, atol=1))

        Cai = sabs_pkpd.run_model.quick_simulate(sabs_pkpd.constants.s,
                                                 1000,
                                                 'calcium.Ca_i',
                                                 pre_run=150000,
                                                 fixed_params_annot=['ical.scale_cal'],
                                                 fixed_params_values=[1])
        Cai = Cai[0]
        CaiD90 = sabs_pkpd.cardiac.compute_calcium_transient_duration(Cai)
        expected_CaiD90 = 384
        assert (np.isclose(CaiD90, expected_CaiD90, atol=2))

        times = np.hstack((np.linspace(0, 250, 1001), np.linspace(251, 1000, 250)))
        Cai = sabs_pkpd.run_model.quick_simulate(sabs_pkpd.constants.s,
                                                 1000,
                                                 'calcium.Ca_i',
                                                 pre_run=150000,
                                                 time_samples=times)
        Cai = Cai[0]
        CaiD90 = sabs_pkpd.cardiac.compute_calcium_transient_duration(Cai, time_points=times)
        expected_CaiD90 = 382
        assert (np.isclose(CaiD90, expected_CaiD90, atol=2))

        CaiD90 = sabs_pkpd.cardiac.compute_calcium_transient_duration(Cai, time_points=times, upstroke_time=50)
        expected_CaiD90 = 389.5
        assert (np.isclose(CaiD90, expected_CaiD90, atol=2))

        # Test exceptions
        with self.assertRaises(ValueError) as context:
            CaiD = sabs_pkpd.cardiac.compute_calcium_transient_duration('not_cai')
        assert 'Cai should be provided' in str(context.exception)

        with self.assertRaises(ValueError) as context:
            CaiD = sabs_pkpd.cardiac.compute_calcium_transient_duration(Cai, time_points='not time points')
        assert 'time_points should be provided' in str(context.exception)

        # Test upstroke time not in time_points
        CaiD90 = sabs_pkpd.cardiac.compute_calcium_transient_duration(Cai, time_points=times, upstroke_time=50.0001)
        expected_CaiD90 = 389.5
        assert (np.isclose(CaiD90, expected_CaiD90, atol=2))


        # Test abnormal Cai
        cai = np.array([-0.01, 10.0])
        CaiD = sabs_pkpd.cardiac.compute_calcium_transient_duration(cai)



        return None
