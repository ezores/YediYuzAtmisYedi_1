```md
# Réseau de Neurones pour la Reconnaissance de la Parole
```
Ce projet implémente un réseau de neurones à rétropropagation du gradient d’erreur pour l'extraction de caractéristiques et la reconnaissance de la parole. L'implémentation est basée sur NumPy pour des performances optimisées.
```
📂 Structure du Projet


/Projet_NN
│── README.md            # Documentation du projet
│── main.py              # Script principal d'entraînement et d'évaluation
│── mlp.py               # Implémentation du réseau de neurones multicouches
│── mlp_math.py          # Fonctions mathématiques (activations, dérivées, etc.)
│── propagation.py       # Propagation avant et arrière du réseau
│── training.py          # Gestion de l'entraînement et de la validation
│── file_utils.py        # Fonctions de gestion des fichiers et logs
│── OutputFiles/         # Dossier contenant les fichiers de données
│── pictures/            # Visualisations du réseau
│── requirements.txt     # Dépendances Python

```
🚀 Installation

1️⃣ Prérequis
- Python 3.8+
- pip installé
``` ```
2️⃣ Cloner le dépôt
```sh
git clone [https://github.com/utilisateur/Projet_NN.git](https://github.com/ezores/YediYuzAtmisYedi_1.git)
cd YediYuzAtmisYedi
```

3️⃣ Installer les dépendances
```sh
pip install -r requirements.txt
```
🎯 Utilisation
Pour démarer le programme, écrivez cette commande dans le terminal:
    -python main.py

Suivit de :
help :pour voir un exemple des possibilitées dans le terminal. LE terminal devrait montrer la ligne suivante:
usage: main.py [-h] (--train | --vc | --test) [--eta ETA] [--neurons NEURONS [NEURONS ...]] [--activations ACTIVATIONS [ACTIVATIONS ...]] [--base {40,50,60}] [--epochs EPOCHS] [--adaptive] [--noise NOISE]
main.py: error: one of the arguments --train --vc --test is required

le premier choix est obligatoire et doit être uniquement l'un des trois choix entre:
--train: pour faire l'entraînement du MLP.
--vc:    pour faire la valiudation croisé du MLP.
--test:  pour tester le MLP.

Si aucun autre paramètre n'est entrée, des paramètres seront demandé à l'utilisateur pour chacune des possibilitées. Pour chaque paramètre deamndé, il est possible d'entrer une valeur désirée
ou de ne rien entrer pour garder la valeur proposé par le terminal.

Entraîner un modèle
L'entraînement d'un modèle peut être lancé avec différents hyperparamètres :
```sh
python main.py --train --eta 0.001 --neurons 128 64 --activations relu --base 60 --epochs 100 --adaptive --noise 0.1
```
🔹 Exemples de paramètres :
| Argument | Description |
|----------|------------|
| `--eta` | Taux d’apprentissage |
| `--neurons` | Nombre de neurones par couche cachée.Exemple: pour 1 couche de 10 neurones et une 2e couche de 5 neurones, ont met (--neurons 10 5) |
| `--activations` | Fonction d'activation (`relu`, `sigmoide`, `tanh`, etc.) |
| `--base` | Base de données utilisée (40, 50 ou 60) |
| `--epochs` | Nombre d'époques |
| `--adaptive` | Active l'apprentissage adaptatif |
| `--noise` | Ajoute un bruit aléatoire aux données

``` ```
🛠️ Personnalisation

Modifier les fonctions d'activation
Les fonctions d'activation se trouvent dans mlp_math.py. 
Exemple : Ajouter une nouvelle activation Leaky ReLU.
```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(a, alpha=0.01):
    return np.where(a > 0, 1, alpha)

activation_functions['leaky_relu'] = (leaky_relu, leaky_relu_derivative)
```
Ensuite, utiliser :
```sh
python main.py --train --activations leaky_relu
```

📊 Visualisation

Les poids et l’architecture du réseau peuvent être visualisés et générés automatiquement après le fonctionnement du code et l'ancien s'efface quand on relance le code.

📁 Les images générées sont stockées dans le dossier `pictures/`. et HTML comme le fichier : `Visualisation_MLP.html` 

🏎️ Optimisations & Performances

🔹 Passage de SymPy à NumPy → Amélioration significative du temps d'exécution.  
🔹 Propagation vectorisée avec `np.dot()` → Accélération des calculs.  
🔹 Application de dropout (uniquement en entraînement) pour limiter le surapprentissage.  
🔹 Normalisation des entrées → Meilleure stabilité du modèle.  

## 📜 Licence

Ce projet est sous licence **MIT**.  
Libre d'utilisation, de modification et de distribution.

🚀 Prêt à entraîner un réseau de neurones performant pour la reconnaissance de la parole !

