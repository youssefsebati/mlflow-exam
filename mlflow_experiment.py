import mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# set the MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow-web:5000")

# create an MLflow experiment
experiment_name = "iris-classification"
mlflow.set_experiment(experiment_name)
mlflow.autolog()

print(f"Starting experiment: {experiment_name}")

# load the iris dataset
print("Loading the iris dataset")
db = load_iris()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target, test_size=0.2, random_state=42)

# create and train a random forest classifier
print("Training a random forest classifier")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# predict on the test set
print("Predicting on the test set")
y_pred = rf.predict(X_test)

print("Experiment complete")