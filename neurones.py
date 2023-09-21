import random
import math

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
                print(liste[i], " x ", self.__coefficients[i], " = ", liste[i] * self.__coefficients[i])
            print("output = ", output)
            output += self.__coefficients[self.__nombre_entrees]
            print(self.__coefficients[self.__nombre_entrees])
            print("output = ", output)
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
    
    def getOutput(self, liste):
        output = super().getOutput(liste)
        return(sigmoid(output))
        print("ta mère")