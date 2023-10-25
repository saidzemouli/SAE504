import random
import matplotlib.pyplot as plt
from neurones import *

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

neurone = SigmoidNeuron(7)
learning = Learning(neurone, entrees, sortie)

erreurs = learning.apprendreSimple(1000)

plt.plot(erreurs)
plt.xlabel('Epochs')
plt.ylabel('Erreur moyenne')
plt.title("Évolution de l'erreur moyenne")
plt.show()
