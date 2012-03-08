from simpl import lp
import numpy as np


class TestLP(object):
    def test_predict(self):
        """test_predict"""
        coefs = np.array([1, 2, 3, 4, 5])
        test_signal = np.ones(5)
        predictions = lp.predict(test_signal, coefs, 2)
        assert predictions[0] == -sum(coefs)
        assert predictions[1] == -sum(coefs[1:]) - predictions[0]
