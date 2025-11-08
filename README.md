# Création d’une API pour la détection et l’analyse des tumeurs cérébrales
**Auteur :** Max89222

Ce projet vise à créer une API utilisant un réseau de neurones convolutif (CNN) grâce au transfer learning et à l'architecture ResNet dans le but de détecter la présence et le type de tumeurs cérébrales

---

## 1) Structure et utilité des fichiers

- **`main.py`** : fichier Python contenant le code source du projet  
- **`app.py`** : serveur Flask contenant notre API de détection de tumeurs du cerveau  
- **`Training/`** : dossier contenant toutes les images du train set  
- **`Testing/`** : dossier contenant toutes les images du test set  
- **`Dockerfile`** : fichier Docker permettant de créer notre image personnalisée pour conteneuriser le projet  
- **`model_save_detec_tumeur_2`** : fichier contenant les paramètres du modèle entraîné, permettant d'utiliser le modèle sans réentraînement  
- **`requirements.txt`** : fichier listant toutes les bibliothèques nécessaires pour lancer le projet  

---

## 2) Dataset utilisé

Ce dataset provient du site Kaggle : [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

**Répartition des labels :**
- glioma : 300  
- meningioma : 306  
- notumor : 405  
- pituitary : 300  

**Taille du dataset :**
- train set : 5712  
- test set : 1311  
- total : 7023  

---

## 3) Résultats et métriques

**Scores obtenus :**

- **Train set :**
   - train accuracy : 0.9278711484593838
   - train precision : 0.9257332323310266
   - train recall : 0.9249702731203867

- **Test set :**
   - test accuracy : 0.9107551487414187
   - test precision : 0.9113846872344679
   - test recall : 0.9045806100217865

**Learning curves :**

<img width="566" height="413" alt="download" src="https://github.com/user-attachments/assets/407e0f46-d304-45cf-969a-0be3bfc95b07" />


**Interprétation :**  
Les résultats sont quasiment similaires sur le train set et le test set. Pas d'overfitting ni d'underfitting : le modèle apprend bien.

---

## 4) Installation et lancement du projet

### Installer Git (si nécessaire) :

`brew install git`

### Cloner le dépot
`git clone <clé_ssh>
cd <nom_du_dossier>`

### Entraîner le modèle (optionnel car celui ci est déjà entraîné) :
`python main.py`
ou
`python3 main.py`

### Construire l'image docker :
`docker build -t detection_tumeur_flask .`

### Lancer le conteneur :
`docker run -it -p 8000:8000 detection_tumeur_flask`

### Effectuer une requête à l'aide de curl ou Postman :
`curl -X POST http://127.0.0.1:8000/ -F "file=@/chemin/vers/votre/fichier"`

### Exemple de résultat renvoyé par l'API : 
`{"résultat":"meningioma"}`

Remarque : pour une API professionnelle fonctionnelle 24h/24, un hébergement sur le cloud est nécessaire
