import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
data = pd.read_csv('Churn_Modelling.csv')

# 2. Data Preprocessing
# Drop unnecessary columns
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encode the categorical columns: 'Gender' and 'Geography'
# Label encode 'Gender' (Binary variable: Male/Female)
labelencoder = LabelEncoder()
data['Gender'] = labelencoder.fit_transform(data['Gender'])

# One-hot encode 'Geography' (Nominal variable: France, Spain, Germany)
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

# 3. Split the dataset into features (X) and target (y)
X = data.drop('Exited', axis=1)  # Features (independent variables)
y = data['Exited']  # Target (dependent variable)

# 4. Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature Scaling
scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Model Training - Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees in the forest
rf_model.fit(X_train, y_train)

# 7. Model Prediction
y_pred = rf_model.predict(X_test)

# 8. Model Evaluation
# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# 9. Feature Importance
importances = rf_model.feature_importances_
features = X.columns
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance)

# Plot feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.show()
