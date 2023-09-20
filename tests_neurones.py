#!/usr/bin/env python
# encoding: utf-8

import unittest
from neurones import Neurone
import random

class TestNeurone(unittest.TestCase):
    def test_return_float_coefficients(self):
        # Vérifie que chaque coefficient est un nombre à virgule flottante (float)
        nombre_entrees = 7
        neurone = Neurone(nombre_entrees)

        coefficients = [neurone.getCoefficient(i) for i in range(nombre_entrees + 1)]

        for coefficient in coefficients:
            self.assertTrue(isinstance(coefficient, float))

    def test_length_coefficients(self):
        # Vérifie que la longueur de la liste coefficients est égale au nombre d'entrées
        nombre_entrees = 7
        neurone = Neurone(nombre_entrees)
        coefficients = [neurone.getCoefficient(i) for i in range(nombre_entrees + 1)]

        self.assertEqual(len(coefficients), nombre_entrees + 1)

    def test_getNeuronSize(self):
        # Test de la méthode pour obtenir le nombre d'entrées
        nombre_entrees = 7
        neurone = Neurone(nombre_entrees)
        self.assertEqual(neurone.getNeuronSize(), nombre_entrees)
    
    def test_Coefficient(self):
        # Test de la méthode pour modifier le coefficient d'index i
        neurone = Neurone(7)
        neurone.setCoefficient(0.5, 0)
        self.assertEqual(neurone.getCoefficient(0), 0.5)
    
    def test_getOutput(self):
        neurone = Neurone(7)
        input_list = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]

        for n in range(neurone.getNeuronSize() + 1):
            neurone.setCoefficient(0, n)
        
        for i in range(neurone.getNeuronSize() + 1):
            neurone.setCoefficient(random.uniform(-1, 1), i)

        calculated_output = neurone.getOutput(input_list)
        expected_output = sum([neurone.getCoefficient(i) * input_list[i] for i in range(len(input_list))])

        self.assertEqual(calculated_output, expected_output)

if __name__ == '__main__':
    unittest.main()
