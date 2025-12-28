import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model import get_model

# Load Configuration
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

print("Config loaded:", config,"\n")  # Debug line

# Load Data
data = pd.read_csv("data/data.csv")

X = data[["feature1","feature2"]]
y = data['target']

# Train Test Split

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=config["training"]["test_size"],
                                                 random_state=config["training"]["random_state"])

# Model fit
model = get_model()
model.fit(X_train,y_train)

# Evaluation
preds = model.predict(X_test)

mse = mean_squared_error(y_test,preds)
print("Mean Squared Error :- ",mse,"\n")
print("Model trained successfully")