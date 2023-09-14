import random

class Neurone:
    def __init__(self, nombre_entrees):
        # Vérifie que le nombre d'entrées est supérieur à zéro
        if nombre_entrees <= 0:
            raise ValueError("Le nombre d'entrées doit être supérieur à zéro.") 
        self.nombre_entrees = nombre_entrees
        # Création de la liste coefficients
        self.coefficients = [random.uniform(-1, 1) for _ in range(nombre_entrees+1)] 
    
    def getNeuronSize(self):
        return (self.nombre_entrees)
