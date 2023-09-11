import random

class Neurone:
    def __init__(self, nombre_entrees): 
        if nombre_entrees <= 0: 
            raise ValueError("Le nombre d'entrées doit être supérieur à zéro.")
        
        self.coefficients = [random.uniform(-1, 1) for _ in range(nombre_entrees)]
        print(self.coefficients) 
