import numpy as np
import sabs_pkpd
import pytest


def test_inputs():
    AP = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    with pytest.raises(ValueError):
        sabs_pkpd.cardiac.compute_APD(AP)

    with pytest.raises(ValueError):
        sabs_pkpd.cardiac.compute_calcium_transient_duration(AP)

    AP = sabs_pkpd.run_model.quick_simulate(sabs_pkpd.constants.s,
                                            1000,
                                            'membrane.V',
                                            pre_run=150000,
                                            fixed_params_annot=['ical.scale_cal'],
                                            fixed_params_values=[0.5],
                                            time_samples=np.linspace(0, 1000, 1001))

    time_points = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    with pytest.raises(ValueError):
        sabs_pkpd.cardiac.compute_APD(AP, time_points)

    with pytest.raises(ValueError):
        sabs_pkpd.cardiac.compute_calcium_transient_duration(AP, time_points)

    return None


def test_compute_APD():
    mmt = './tests/test resources/tentusscher_2006_pints_and_Chons_hERG.mmt'
    sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(mmt)

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

    return None


def test_compute_calcium_transient_duration():
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

    return None
