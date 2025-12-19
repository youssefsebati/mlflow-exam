import mlflow
import numpy as np

# set the MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow-web:5000")

# load the model
model_uri = "models:/iris/Production"
model = mlflow.pyfunc.load_model(model_uri)

# create a sample input
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

# predict using the model
prediction = model.predict(sample_input)

print(f"Sample Input: {sample_input}")
print(f"Prediction: {prediction}")