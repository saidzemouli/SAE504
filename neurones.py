import random

class Neurone:

    def __init__(self, nombre_entrees):
        if not isinstance(nombre_entrees, int):
            raise ValueError("Le nombre d'entrées doit être un nombre entier.")
        elif nombre_entrees <= 0:
            raise ValueError("Le nombre d'entrées doit être supérieur à zéro.")
        else:
            self.nombre_entrees = nombre_entrees
            # Création de la liste coefficients
            self.coefficients = [random.uniform(-1, 1) for _ in range(nombre_entrees + 1)]

    def getNeuronSize(self): # Méthode pour obtenir le nombre d'entrées
        return (self.nombre_entrees)
    
    def getCoefficient(self, index):
        if not isinstance(index, int):
            raise ValueError("L'index doit être un nombre entier.")
        if index < 0 or index > self.nombre_entrees:
            raise ValueError("L'index doit être compris entre 0 et le nombre d'entrées.")
        else:
            return (self.coefficients[index])
        
    def setCoefficient(self,valeur,index):
        if not isinstance(index, int):
            raise ValueError("L'index doit être un nombre entier.")
        if index < 0 or index > self.nombre_entrees:
            raise ValueError("L'index doit être compris entre 0 et le nombre d'entrées.")
        else:
            self.coefficients[index] = valeur

    def getOutput(self,list):
        output = 0
        for i in range(self.nombre_entrees):
            output += list[i]*self.coefficients[i]
            print(list[i]," x ", self.coefficients[i], " = ", list[i]*self.coefficients[i])
        print("output = ", output)
        return output
            