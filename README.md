Projet MLOps : Gestion du cycle de vie ML avec MLflow et Docker


## Structure des fichiers

docker-compose.yaml : Orchestration des services (MLflow, Postgres, MinIO).

dockerfile.mlflow : Image serveur MLflow avec drivers psycopg2 et boto3.

dockerfile.python : Environnement de Data Science (Scikit-learn, MLflow).

mlflow_experiment.py : Script d'entraînement avec autolog().

mlflow_pred.py : Script d'inférence chargeant le modèle depuis le Model Registry.

.gitignore : Exclusion des volumes Docker et caches Python pour un dépôt propre.



Ce projet a été réalisé dans le cadre de l'examen de l'ISC M2. Il présente une infrastructure complète pour le suivi d'expériences, le stockage de modèles et la reproductibilité des environnements de Machine Learning.

## Architecture du Système

L'infrastructure est entièrement conteneurisée avec **Docker Compose** et repose sur trois piliers :

1.  **MLflow Tracking Server :** Centralise le suivi des métriques, paramètres et versions de modèle
2.  **PostgreSQL (Backend Store) :** Base de données relationnelle stockant les métadonnées des expériences
3.  **MinIO (Artifact Store) :** Serveur de stockage d'objets (S3 compatible) où sont sauvegardés les modèles entraînés (fichiers `.pkl`, fichiers de configuration)


## Guide

### 1

Dans terminal, exécutez la commande suivante pour démarrer l'infrastructure :

docker compose up -d

### 2
Configuration de l'Artifact Store (MinIO)

**Créer le bucket de stockage :**

Go sur : http://localhost:8900
Identifiants : mlflow / mlflow123
Cliquez sur Buckets > Create Bucket
Nommez le bucket : mlflow

### 3

Execution de l'experience

# Construction de l'image de l'environnement Python
docker build -t dockerfile.python -f dockerfile.python .

# Lancement de l'expérience
docker run --rm -v "%cd%:/app" -w /app --network mlflow-network \
    --env AWS_ACCESS_KEY_ID=mlflow \
    --env AWS_SECRET_ACCESS_KEY=mlflow123 \
    --env MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
    dockerfile.python python mlflow_experiment.py

Une fois le script terminé, les résultats sont accessibles sur les interfaces suivantes :

MLflow UI : http://localhost:5000
Consultez l'expérience iris-classification
Visualisez les métriques (accuracy), les paramètres (n_estimators) et téléchargez le modèle dans la section Artifacts

MinIO Browser : http://localhost:8900
Vérifiez la persistance des fichiers du modèle dans le bucket mlflow
