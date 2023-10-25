import random
import math
import matplotlib.pyplot as plt

class Neurone:

    def __init__(self, nombre_entrees):
        if not isinstance(nombre_entrees, int):
            raise ValueError("Le nombre d'entrées doit être un nombre entier.")
        elif nombre_entrees <= 0:
            raise ValueError("Le nombre d'entrées doit être supérieur à zéro.")
        else:
            self.__nombre_entrees = nombre_entrees
            # Création de la liste coefficients
            self.__coefficients = [random.uniform(-1, 1) for _ in range(nombre_entrees + 1)]

    def getNeuronSize(self): # Méthode pour obtenir le nombre d'entrées
        return self.__nombre_entrees
    
    def getCoefficient(self, index):
        if not isinstance(index, int):
            raise ValueError("L'index doit être un nombre entier.")
        if index < 0 or index > self.__nombre_entrees:
            raise ValueError("L'index doit être compris entre 0 et le nombre d'entrées.")
        else:
            return self.__coefficients[index]
        
    def setCoefficient(self, valeur, index):
        if not isinstance(index, int):
            raise ValueError("L'index doit être un nombre entier.")
        if index < 0 or index > self.__nombre_entrees:
            raise ValueError("L'index doit être compris entre 0 et le nombre d'entrées.")
        else:
            self.__coefficients[index] = valeur

    def getOutput(self, liste):
        output = 0
        if not isinstance(liste, list):
            raise ValueError("La liste doit être une liste.")
        if len(liste) != self.__nombre_entrees:
            raise ValueError("La liste doit avoir le même nombre d'entrées que le neurone.")
        if not all(isinstance(x, float) for x in liste):
            raise ValueError("Les valeurs de la liste doivent être des nombres à virgule flottante (float).")
        else:
            for i in range(self.__nombre_entrees):
                output += liste[i] * self.__coefficients[i]
                #print(liste[i], " x ", self.__coefficients[i], " = ", liste[i] * self.__coefficients[i])
            #print("output = ", output)
            output += self.__coefficients[self.__nombre_entrees]
            #print(self.__coefficients[self.__nombre_entrees])
            #print("output = ", output)
            return output

def sigmoid(x):
    if x > 100 :
        return(1.0)
    elif x < -100 :
        return(0.0)
    else :
        return(1.0 / (1.0 + math.exp(-x)))
    
class SigmoidNeuron(Neurone):
    def __init__(self, nombre_entrees):
        super().__init__(nombre_entrees)
    
    def calculeSortie(self, liste):
        return sigmoid(super().getOutput(liste))
    

class Learning:
    def __init__(self, neurone, entrees, sortie):
        self.neurone = neurone
        self.entrees = entrees
        self.sortie = [sortie]

    def calculerErreur(self, entree, sortie):
            prediction = self.neurone.calculeSortie(entree)
            erreur = (sortie - prediction) ** 2
            return erreur

    def calculerErreurMoyenne(self):
        total_erreur = 0
        for i in range(len(self.entrees)):
            total_erreur += self.calculerErreur(self.entrees[i], self.sortie[i])
        erreur_moyenne = total_erreur / len(self.entrees)
        return erreur_moyenne
    
    def apprendreSimple(self, epochs=1000):
        erreurs = []

        for epoch in range(epochs):
            erreur_actuelle = self.calculerErreurMoyenne()
            erreurs.append(erreur_actuelle)

            # Choisir un coefficient au hasard dans le neurone
            coefficient_index = random.randint(0, self.neurone.getNeuronSize())
            ancienne_valeur = self.neurone.getCoefficient(coefficient_index)

            # Ajoutez une valeur aléatoire entre -0.1 et +0.1 au coefficient
            delta = random.uniform(-0.1, 0.1)
            self.neurone.setCoefficient(ancienne_valeur + delta, coefficient_index)

            # Recalcule del'erreur moyenne
            nouvelle_erreur = self.calculerErreurMoyenne()

            # Si la nouvelle erreur est plus grande, remettre l'ancienne valeur du coefficient
            if nouvelle_erreur > erreur_actuelle:
                self.neurone.setCoefficient(ancienne_valeur, coefficient_index)

        return erreurs
    
 