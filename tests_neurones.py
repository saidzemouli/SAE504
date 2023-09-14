#!/usr/bin/env python
# encoding: utf-8

import unittest
from neurones import *

class TestNeurone(unittest.TestCase):
    def test_return_float_coefficients(self): # Vérifie que chaque coefficient est un nombre à virgule flottante (float)

        # Créez une instance de Neurone avec un nombre arbitraire d'entrées
        nombre_entrees = 7
        neurone = Neurone(nombre_entrees)

        # Accédez aux coefficients à partir de l'instance de Neurone
        coefficients = neurone.coefficients

        # Vérifiez que chaque coefficient est un nombre à virgule flottante (float)
        for coefficient in coefficients:
            self.assertTrue(isinstance(coefficient, float))

    def test_length_coefficients(self): # Vérifie que la longueur de la liste coefficients est égale au nombre d'entrées

        nombre_entrees = 7
        neurone = Neurone(nombre_entrees)
        coefficients = neurone.coefficients
        # Vérifie que la longueur de la liste coefficients est égale au nombre d'entrées
        self.assertEqual(len(coefficients), nombre_entrees+1)

    def test_getNeuronSize(self): # Test de la méthode pour obtenir le nombre d'entrées
        nombre_entrees = 7
        neurone = Neurone(nombre_entrees)
        self.assertEqual(neurone.getNeuronSize(), nombre_entrees)
    
    def test_getCoefficient(self): # Test de la méthode pour obtenir le coefficient d'index i
        neurone = Neurone(7)
        self.assertEqual(neurone.getCoefficient(0), neurone.coefficients[0])

if __name__ == '__main__':
    unittest.main()