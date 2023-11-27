import random
import matplotlib.pyplot as plt
from neurones import *


# Données d'entrée et de sortie
entrees = [
    [1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1]
]

sortie = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

# Créer le neurone et l'objet Learning
neurone = SigmoidNeuron(7)
learning = Learning(neurone, entrees, sortie)

# Apprentissage
erreurs = learning.apprendreSimple(1000)

# Affichage de l'évolution de l'erreur
plt.plot(erreurs)
plt.xlabel('Epochs')
plt.ylabel('Erreur moyenne')
plt.title("Évolution de l'erreur moyenne")
plt.show()

# Vérification de la qualité de l'apprentissage avec des données bruitées
entrees_bruitees = [[x + random.uniform(-0.1, 0.1) for x in entree] for entree in entrees]
resultats_bruites = [neurone.calculerSortie(entree) > 0.5 for entree in entrees_bruitees]

print(resultats_bruites)