import math
import random
import matplotlib.pyplot as plt
import numpy as np

class Neurone:
    def __init__(self, nombre_entrees):
        if nombre_entrees <= 0:
            raise ValueError("Le nombre d'entrées doit être supérieur à zéro.")
        
        # Initialisation aléatoire des coefficients entre -1 et 1
        self.coefficients = [random.uniform(-1, 1) for _ in range(nombre_entrees + 1)]

    def getNeuronSize(self):
        # Nombre d'entrées
        return len(self.coefficients) - 1
    
    def getCoefficient(self, position):
        # Retourne la valeur du coefficient à la position donnée
        if 0 <= position < len(self.coefficients):
            return self.coefficients[position]
        else:
            raise ValueError("Position de coefficient invalide.")

    def setCoefficient(self, position, valeur):
        # Modifie la valeur du coefficient à la position donnée
        if 0 <= position < len(self.coefficients):
            self.coefficients[position] = valeur
        else:
            raise ValueError("Position de coefficient invalide.")
        
    def getOutput(self, entrees):
        # Calcule la sortie du neurone pour la configuration donnée.
        if len(entrees) != len(self.coefficients) - 1:
            raise ValueError("Le nombre d'entrées ne correspond pas à la taille du neurone.")
        
        somme_ponderee = sum(entrees[i] * self.coefficients[i] for i in range(len(entrees)))
        return somme_ponderee

def sigmoid(a):
    if a > 100:
        return 1.0
    if a < -100:
        return 0.0
    else:
        return 1 / (1 + math.exp(-a))

class SigmoidNeuron(Neurone):
    def __init__(self, entre):
        Neurone.__init__(self, entre)

    def calculerSortie(self, num):
        return sigmoid(super().getOutput(num))
                    
class Learning:
    def __init__(self, neurone, entrees, sortie):
        if len(entrees) != len(sortie):
            raise ValueError("Le nombre d'entrées ne correspond pas au nombre de sortie.")
        for i in range(len(entrees)): 
            if len(entrees[i]) != neurone.getNeuronSize():
                raise ValueError("Le nombre d'entrées ne correspond pas à la taille du neurone.")
        self.neurone = neurone
        self.entrees = entrees
        self.sortie = sortie

    def calculerErreur(self, index):
        # Calculer la sortie du neurone pour le jeu d'entrées 'entrees[index]'.
        neurone_sortie = self.neurone.calculerSortie(self.entrees[index])
        # Calculer l'erreur en carré.
        erreur = (neurone_sortie - self.sortie[index]) ** 2
        return erreur

    def calculerErreurMoyenne(self):
        # Calculer la moyenne des erreurs pour tous les éléments de 'entrees' et 'sortie'.
        total_erreur = sum(self.calculerErreur(index) for index in range(len(self.entrees)))
        erreur_moyenne = total_erreur / len(self.entrees)
        return erreur_moyenne

    def apprendreSimple(self, epochs=1000):
        erreurs = []  # Stocker l'évolution de l'erreur moyenne.
        for _ in range(epochs):
            erreur_actuelle = self.calculerErreurMoyenne()
            erreurs.append(erreur_actuelle)
            # Choisir un coefficient au hasard dans le neurone.
            random_coefficient_index = random.randint(0, len(self.neurone.coefficients) - 1)
            old_coefficient = self.neurone.coefficients[random_coefficient_index]
            # Ajouter une valeur aléatoire entre -0.1 et +0.1.
            random_change = random.uniform(-0.1, 0.1)
            new_coefficient = old_coefficient + random_change
            # Mettre à jour le coefficient dans le neurone.
            self.neurone.coefficients[random_coefficient_index] = new_coefficient
            # Recalculer l'erreur moyenne.
            nouvelle_erreur = self.calculerErreurMoyenne()
            # Si la nouvelle erreur est plus élevée, remettre l'ancienne valeur au coefficient.
            if nouvelle_erreur > erreur_actuelle:
                self.neurone.coefficients[random_coefficient_index] = old_coefficient

        return erreurs
    
    def apprendreAvecMemoire(self, epochs=1000):
        erreurs = []  # Stocker l'évolution de l'erreur moyenne.
        coefficients_memoire = []  # Stocker les coefficients qui ont été utiles pour la baisse de l'erreur.

        for _ in range(epochs):
            erreur_actuelle = self.calculerErreurMoyenne()
            erreurs.append(erreur_actuelle)

            # Choisir un coefficient au hasard dans le neurone.
            random_coefficient_index = random.randint(0, len(self.neurone.coefficients) - 1)
            old_coefficient = self.neurone.coefficients[random_coefficient_index]

            # Choisir une valeur entre 0 et 0.25 et un signe (True pour positif, False pour négatif).
            random_value = random.uniform(0, 0.25) # J'ai modifié la valeur de 0.1 à 0.25 pour que l'erreur diminue plus rapidement.
            random_sign = random.choice([True, False])
            modification = random_value if random_sign else -random_value

            # Stocker la valeur du coefficient avant la modification.
            coefficients_memoire.append(old_coefficient)

            # Changer la valeur du coefficient selon la modification et le signe.
            self.neurone.coefficients[random_coefficient_index] += modification

            # Recalculer l'erreur moyenne.
            nouvelle_erreur = self.calculerErreurMoyenne()

            # Comparer les erreurs et ajuster la modification.
            if nouvelle_erreur >= erreur_actuelle:
                # Si la nouvelle erreur est plus élevée ou égale, revenir à la valeur précédente.
                self.neurone.coefficients[random_coefficient_index] = old_coefficient
            else:
                # Si la nouvelle erreur est plus faible, augmenter la valeur absolue de la modification.
                modification *= 1.1

        return erreurs, coefficients_memoire

class NeuralNetwork:
    def __init__(self, num_inputs, layers, neuron_counts, neuron_types):
        if len(layers) != len(neuron_counts) or len(layers) != len(neuron_types):
            raise ValueError("Inconsistent parameters for layers, neuron_counts, and neuron_types.")

        valid_neuron_types = ["Neurone", "SigmoidNeurone"]  # Updated neuron types
        if not all(type_neurone in valid_neuron_types for type_neurone in neuron_types):
            raise ValueError(f"Invalid neuron type. Must be one of {valid_neuron_types}.")

        self.layers = []

        # Create layers with neurons
        for i in range(len(layers)):
            layer = []
            num_inputs_layer = num_inputs if i == 0 else neuron_counts[i - 1]
            neuron_type = Neurone if neuron_types[i] == "Neurone" else SigmoidNeuron  # Updated neuron type

            for _ in range(neuron_counts[i]):
                layer.append(neuron_type(num_inputs_layer))

            self.layers.append(layer)

    def get_coefficient(self, layer, neuron, position):
        if 0 <= layer < len(self.layers) and 0 <= neuron < len(self.layers[layer]):
            return self.layers[layer][neuron].getCoefficient(position)
        else:
            raise ValueError("Couche ou position de neurone invalide.")

        
