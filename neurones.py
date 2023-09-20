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

    def getOutput(self,liste):
        output = 0
        if not isinstance(liste, list):
            raise ValueError("La liste doit être une liste.")
        if len(liste) != self.nombre_entrees:
            raise ValueError("La liste doit avoir le même nombre d'entrées que le neurone.")
        if not all(isinstance(x, (float)) for x in liste):
            raise ValueError("Les valeurs de la liste doivent être des nombres à virgule flottante (float).")
        else :

            for i in range(self.nombre_entrees):
                output += liste[i]*self.coefficients[i]
                print(liste[i]," x ", self.coefficients[i], " = ", liste[i]*self.coefficients[i])
            print("output = ", output)
            return output
            