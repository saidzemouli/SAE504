#!/usr/bin/env python
# encoding: utf-8

import unittest
from neurones import Neurone

class TestNeurone(unittest.TestCase):
    
    def test_coefficients(self):
        neurone = Neurone(7)
        coefficient = neurone.coefficients[0]
        is_float = isinstance(coefficient, float)
        self.assertTrue(is_float)

if __name__ == '__main__':
    unittest.main()
