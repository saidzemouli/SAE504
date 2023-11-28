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

# Sauvegarder les coefficients initiaux
coefficients_initiaux = [neurone.getCoefficient(i) for i in range(neurone.getNeuronSize() + 1)]

# Affichage des coefficients initiaux
print("Coefficients initiaux:", coefficients_initiaux)

# Réaliser l'apprentissage avec apprendreSimple
learning_simple = Learning(neurone, entrees, sortie)
erreurs_simple = learning_simple.apprendreSimple(1000)

# Réinitialiser les coefficients avec les valeurs initiales
for i in range(neurone.getNeuronSize() + 1):
    neurone.setCoefficient(i, coefficients_initiaux[i])

# Réaliser l'apprentissage avec apprendreAvecMemoire
learning_memoire = Learning(neurone, entrees, sortie)
erreurs_memoire, _ = learning_memoire.apprendreAvecMemoire(1000)

# Affichage de l'évolution de l'erreur
plt.plot(erreurs_simple, label="Apprendre Simple", color="red")
plt.plot(erreurs_memoire, label="Apprendre Avec Mémoire", color="blue")
plt.xlabel('Epochs')
plt.ylabel('Erreur moyenne')
plt.title("Évolution de l'erreur moyenne")
plt.legend()
plt.show()
