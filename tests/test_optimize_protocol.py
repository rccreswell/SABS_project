import math
import os
import numpy as np
import sabs_pkpd
from sabs_pkpd.optimize_protocol import ProtocolOptimizer
import unittest

def simple_solver(alpha, protocol, times, x0):
    return np.ones(len(times)) * alpha

def simple_protocol(amplitude, duration):
    return lambda times: np.array(((times > 1.0) & (times < 1.0 + duration)))\
                                    .astype(float) * amplitude

def fake_protocol(amplitude, duration):
    return lambda times: np.zeros(len(times))

class Test(unittest.TestCase):
    def test_init_optimizer(self):
        """Test initializing the optimizer, both with and without true params.
        """

        opt1 = ProtocolOptimizer(
            simple_solver,
            simple_protocol,
            np.linspace(0, 10, 25),
            1.0,
            [1.5],
            [0.5, 0.5],
            100,
            true_model_params=[1.5])

        assert opt1.true_model_params == [1.5]
        assert np.allclose(opt1.times, np.linspace(0, 10, 25))
        assert np.allclose(opt1.model_params, [1.5])
        assert np.allclose(opt1.protocol_params, [0.5, 0.5])

        opt2 = ProtocolOptimizer(
            simple_solver,
            simple_protocol,
            np.linspace(0, 10, 25),
            1.0,
            [1.5],
            [0.5, 0.5],
            100)

        assert opt2.true_model_params == [1.5]
        assert np.allclose(opt2.times, np.linspace(0, 10, 25))
        assert np.allclose(opt2.model_params, [1.5])
        assert np.allclose(opt2.protocol_params, [0.5, 0.5])

    def test_run_mcmc(self):
        """Test running MCMC with the original protocol.
        """
        opt = ProtocolOptimizer(
            simple_solver,
            simple_protocol,
            np.linspace(0, 10, 25),
            1.0,
            [1.5],
            [0.5, 0.5],
            100,
            true_model_params=[1.5])

        opt.run_original_protocol()

        # Check all the expected results of running
        assert np.allclose(opt.original_protocol_params, [0.5, 0.5])
        assert opt.original_data.shape[0] > 0
        assert opt.original_posterior.shape[0] > 0

    def test_optimize_protocol(self):
        """Test running the protocol optimizer routine.
        """

        # Test without parallel mode
        opt = ProtocolOptimizer(
            simple_solver,
            simple_protocol,
            np.linspace(0, 10, 25),
            1.0,
            [1.5],
            [0.5, 0.5],
            100,
            true_model_params=[1.5],
            parallel=False)

        opt.run_original_protocol()
        opt.optimize_protocol(max_iters=2)

        assert len(opt.protocol_params) == 2

        # Test with parallel mode
        opt = ProtocolOptimizer(
            simple_solver,
            simple_protocol,
            np.linspace(0, 10, 25),
            1.0,
            [1.5],
            [0.5, 0.5],
            100,
            true_model_params=[1.5],
            parallel=True)

        opt.run_original_protocol()
        opt.optimize_protocol(max_iters=2)

        assert len(opt.protocol_params) == 2

        # Test singular matrix sets the objective manually
        opt = ProtocolOptimizer(
            simple_solver,
            fake_protocol,
            np.linspace(0, 10, 25),
            1.0,
            [1.5],
            [0.5, 0.5],
            100,
            true_model_params=[1.5],
            parallel=True)

        opt.run_original_protocol()
        opt.optimize_protocol(max_iters=2)

        assert len(opt.protocol_params) == 2


    def test_infer_model_params(self):
        """Test inferring the model parameters.
        """
        opt = ProtocolOptimizer(
            simple_solver,
            simple_protocol,
            np.linspace(0, 10, 25),
            1.0,
            [1.5],
            [0.5, 0.5],
            100,
            true_model_params=[1.5])

        opt.infer_model_parameters()
        assert opt.data.shape[0] > 0
        assert opt.posterior.shape[0] > 0

    def test_update_model_params(self):
        """Test updating the model parameters.
        """
        opt = ProtocolOptimizer(
            simple_solver,
            simple_protocol,
            np.linspace(0, 10, 25),
            1.0,
            [1.5],
            [0.5, 0.5],
            100,
            true_model_params=[1.5])

        opt.infer_model_parameters()
        opt.update_model_parameters()

    def test_plot(self):
        """Test the plotting function.
        """
        opt = ProtocolOptimizer(
            simple_solver,
            simple_protocol,
            np.linspace(0, 10, 25),
            1.0,
            [1.5],
            [0.5, 0.5],
            100,
            true_model_params=[1.5])

        opt.run_original_protocol()
        opt.infer_model_parameters()
        fig = opt.plot(show=False)
        assert len(fig.axes) > 0

    def test_oscillator(self):
         sim = sabs_pkpd.optimize_protocol.damped_harmonic_oscillator
         protocol = simple_protocol(0.5, 0.5)
         t = np.linspace(0, 10, 100)
         x0 = 0.5
         ts = sim(1.0, 0.5, 0.5, 0.25, protocol, t, x0)
         assert len(ts) == len(t)

if __name__ == '__main__':
    unittest.main()
    # test_init_optimizer()
    # test_run_mcmc()
    # test_optimize_protocol()
    # test_infer_model_params()
