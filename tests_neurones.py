import unittest
from neurones import *
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
        neurone.setCoefficient(0, 0.5)
        self.assertEqual(neurone.getCoefficient(0), 0.5)
    
    def test_getOutput_0(self): # Test de la méthode pour obtenir la sortie du neurone pour coefficient = 0
        neurone = Neurone(7)
        input_list = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]

        for n in range(neurone.getNeuronSize() + 1):
            neurone.setCoefficient(n, 0)
    
        calculated_output = neurone.getOutput(input_list)
        expected_output = 0

        self.assertEqual(calculated_output, expected_output)

    def test_getOutput_result(self): # Test de la méthode pour obtenir la sortie du neurone pour coefficient aléatoire
        neurone = Neurone(7)
        input_list = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]

        for i in range(neurone.getNeuronSize() + 1):
            neurone.setCoefficient(i, random.uniform(-1, 1))
        
        calculated_output = neurone.getOutput(input_list)
        expected_output = sum([neurone.getCoefficient(i) * input_list[i] for i in range(len(input_list))])

        self.assertEqual(calculated_output, expected_output)

  
    def test_sigmoid(self): # Test de la fonction sigmoid
        self.assertEqual(sigmoid(-150), 0)
        self.assertEqual(sigmoid(150), 1.0)
        self.assertEqual(sigmoid(0), 0.5)
        self.assertLess(sigmoid(-10), 0.25)
        self.assertGreater(sigmoid(10), 0.75)

    def test_calculeSortie_coeff0(self): # Test de la méthode calculeSortie pour obtenir la sortie du neurone pour coefficient = 0
        neurone = SigmoidNeuron(7)
        input_list = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]

        for n in range(neurone.getNeuronSize() + 1):
            neurone.setCoefficient(n, 0)
        self.assertEqual(neurone.calculerSortie(input_list), 0.5)
    
    def test_calculeSortie_coeff_positif(self):
        neurone = SigmoidNeuron(7)
        input_list = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]

        for n in range(neurone.getNeuronSize() + 1):
            neurone.setCoefficient(n, 1)
        self.assertGreater(neurone.calculerSortie(input_list), 0)
   
    def test_calculeSortie_coeff_negatif(self):
        neurone = SigmoidNeuron(7)
        input_list = [-1.1, -2.2, -3.3, -4.4, -5.5, -6.6, -7.7]

        for n in range(neurone.getNeuronSize() + 1):
            neurone.setCoefficient(n, -1)
        self.assertGreater(neurone.calculerSortie(input_list), 0)

    def test_coherence_coefficients(self):
        # Vérifie que tous les coefficients dans le réseau sont cohérents
        inputs_length = 3
        layers = [2, 3, 1]
        layer_types = ["LIN", "LIN", "LIN"]

        neural_network = NeuralNetwork(inputs_length, layers, layer_types)

        for layer_idx, layer in enumerate(neural_network.layers):
            for neuron_idx, neuron in enumerate(layer):
                for coeff_idx in range(neuron.getNeuronSize() + 1):
                    coefficient = neural_network.get_coefficient(layer_idx, neuron_idx, coeff_idx)
                    self.assertIsInstance(coefficient, float)

if __name__ == '__main__':
    unittest.main()
