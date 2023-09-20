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

    def test_setCoefficient(self): # Test de la méthode pour modifier le coefficient d'index i
        neurone = Neurone(7)
        neurone.setCoefficient(0.5,0)
        self.assertEqual(neurone.getCoefficient(0), 0.5)
    
    def test_getOutput(self):
        neurone = Neurone(7)
        list = [1.1,2.2,3.3,4.4,5.5,6.6,7.7]

        for n in range(neurone.getNeuronSize()+1):
            neurone.setCoefficient(0,n)
        print("Coefficients à 0",neurone.coefficients)
        for i in range (len(neurone.coefficients)):
            neurone.setCoefficient(random.uniform(-1, 1),i)
        print("Coefficients aléatoires",neurone.coefficients)
        neurone.getOutput(list)
        self.assertEqual(neurone.getOutput(list), neurone.coefficients[0]*list[0]+neurone.coefficients[1]*list[1]+neurone.coefficients[2]*list[2]+neurone.coefficients[3]*list[3]+neurone.coefficients[4]*list[4]+neurone.coefficients[5]*list[5]+neurone.coefficients[6]*list[6])

if __name__ == '__main__':
    unittest.main()