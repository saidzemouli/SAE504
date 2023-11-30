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

    def test_get_coefficient(self):
        # Initialisation d'un réseau neuronal pour les tests
        num_inputs = 3
        layers = [1]
        neuron_counts = [1]
        neuron_types = ["Neurone"]

        nn = NeuralNetwork(num_inputs, layers, neuron_counts, neuron_types)

        # Récupération d'un coefficient
        coefficient = nn.get_coefficient(0, 0, 0)

        # Vérification du résultat
        self.assertIsInstance(coefficient, float)  # Vérifie que le coefficient est de type float
        self.assertGreaterEqual(coefficient, -1.0)  # Vérifie que le coefficient est supérieur ou égal à -1.0
        self.assertLessEqual(coefficient, 1.0)  # Vérifie que le coefficient est inférieur ou égal à 1.0

    def test_set_coefficient(self):
        num_inputs = 3
        layers = [1]
        neuron_counts = [1]
        neuron_types = ["Neurone"]

        nn = NeuralNetwork(num_inputs, layers, neuron_counts, neuron_types)
        nn.set_coefficient(0, 0, 0, 0.5)
        self.assertEqual(nn.get_coefficient(0, 0, 0), 0.5)

    class TestNeuralNetwork(unittest.TestCase):
        def test_get_outputs(self):
            num_inputs = 3
            layers = [2, 1]
            neuron_counts = [3, 1]
            neuron_types = ["Neurone", "Neurone"]

            nn = NeuralNetwork(num_inputs, layers, neuron_counts, neuron_types)

            # Définir les coefficients à des valeurs connues pour faciliter le test
            nn.set_coefficient(0, 0, 0, 0.1)
            nn.set_coefficient(0, 0, 1, 0.2)
            nn.set_coefficient(0, 0, 2, 0.3)
            nn.set_coefficient(0, 1, 0, -0.4)
            nn.set_coefficient(0, 1, 1, -0.5)
            nn.set_coefficient(0, 1, 2, -0.6)
            nn.set_coefficient(0, 2, 0, 0.7)
            nn.set_coefficient(0, 2, 1, 0.8)
            nn.set_coefficient(0, 2, 2, 0.9)

            nn.set_coefficient(1, 0, 0, 1.0)
            nn.set_coefficient(1, 0, 1, -1.1)
            nn.set_coefficient(1, 0, 2, 1.2)

            # Entrées arbitraires
            inputs = [0.5, -0.6, 0.7]

            # Attendu : sortie de la première couche (avant activation)
            expected_layer1_output = [0.1*0.5 + 0.2*(-0.6) + 0.3*0.7, -0.4*0.5 - 0.5*(-0.6) - 0.6*0.7, 0.7*0.5 + 0.8*(-0.6) + 0.9*0.7]

            # Attendu : sortie finale (après activation)
            expected_final_output = [sigmoid(1.0*expected_layer1_output[0] - 1.1*expected_layer1_output[1] + 1.2*expected_layer1_output[2])]

            # Test
            outputs = nn.get_outputs(inputs)

            # Vérification des résultats
            self.assertEqual(len(outputs), len(layers))
            self.assertEqual(len(outputs[0]), neuron_counts[0])
            self.assertEqual(len(outputs[-1]), neuron_counts[-1])

            # Vérification de la sortie de la première couche
            self.assertEqual(outputs[0], expected_layer1_output)

            # Vérification de la sortie finale
            self.assertAlmostEqual(outputs[-1][0], expected_final_output[0], places=5)


if __name__ == '__main__':
    unittest.main()