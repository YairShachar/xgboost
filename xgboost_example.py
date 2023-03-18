import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
print("yyy1")
iris = datasets.load_iris()
print("yyy2")
X, y = iris.data, iris.target
print("yyy3")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("yyy4")

# Convert the data into DMatrix format, which is the internal data structure used by XGBoost
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'multi:softprob',  # Multiclass classification
    'num_class': 3,                 # Number of classes in the Iris dataset
    'max_depth': 3,                 # Maximum depth of the decision trees
    'eta': 0.3,                     # Learning rate
}

# Train the XGBoost model
num_rounds = 50
model = xgb.train(params, D_train, num_rounds)

# Make predictions on the test set
y_pred_prob = model.predict(D_test)
y_pred = y_pred_prob.argmax(axis=1)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

