import random

class Neurone:
    
    def setCoefficient(self, nombre_entrees):
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
