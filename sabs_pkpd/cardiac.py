import numpy as np

def compute_APD(AP, time_points=None, upstroke_time=None, repol_percentage=90, print_warnings=True):
    """
    Computes the action potential duration at repol_percentage % of repolarisation for AP.

    :param AP:
    1-D list or 1D numpy.array. List of voltage points for ONE action potential.

    :param time_points:
    1-D list or 1D numpy.array. List of the times at which the voltage is recorded for AP. If not specified, the time
    points are considered to be spaced by 1 ms. The times should be provided in ms.

    :param upstroke_time:
    float. Time at which the AP is triggered .The upstroke will be computed as the point with the maximal dV/dt if not
    specified. Be careful if you are not using linearly spaced time points for recording AP.

    :param repol_percentage:

    :return: APD
    float. Action potential duration at repol_percentage % of repolarisation. It is computed by searching for the first
    AP point (at least 20 ms after upstroke) reaching a voltage verifying: V < min(AP) + (max(AP) - min(AP))*repol_percentage.
    The upstroke will be computed as the point with the maximal dV/dt if not provided.
    """
    # Verify that the inputs are provided as they should
    if type(AP) == list:
        AP = np.array(AP)
    elif type(AP) == np.ndarray:
        if len(np.shape(AP)) > 1:
            raise ValueError(
                'AP should be provided either as a 1-D list or as a numpy array. Here, it was provided as a'
                + np.shape(AP) + ' array.')
    elif type(AP) != np.ndarray:
        raise ValueError('AP should be provided either as a 1-D list or as a numpy array. Type of AP : ' + type(AP))

    # Define baseline as minimal voltage
    max_AP = np.max(AP)
    min_AP = np.min(AP)
    repol_voltage = min_AP + (max_AP - min_AP) * (100 - repol_percentage) / 100

    if min_AP >= -50 or min_AP <= -120 or max_AP < -40 or max_AP > 50 and print_warnings == True:
        print('This AP may be abnormal, baseline is at ' + min_AP + ' mV. Calculating the APD anyway...')

    # Define time_points if not provided. It is assumed that the model is paced at 1 Hz
    if time_points is None:
        time_points = np.linspace(0, 1000, len(AP))

    # Search the index in the time points at which the action potential is triggered
    if upstroke_time is None:
        dVdt = []
        for i in range(len(AP) - 1):
            dVdt.append((AP[i + 1] - AP[i]) / (time_points[i + 1] - time_points[i]))
        upstroke_index = np.where(dVdt == np.max(dVdt))[0]

    elif upstroke_time not in list(time_points):
        upstroke_index = np.where(time_points > upstroke_time)[0]

    else:
        upstroke_index = np.where(time_points == upstroke_time)[0]

    # Search for the APD.
    APD_index = np.where(AP < repol_voltage)
    found_APD = False
    for i in range(len(APD_index[0])):
        if time_points[APD_index[0][i]] > time_points[upstroke_index[0]] + 20 and found_APD == False:
            APD = time_points[APD_index[0][i]] - time_points[upstroke_index[0]]
            found_APD = True
            break

    if found_APD == False:
        APD = 0

    return APD