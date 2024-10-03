import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def preprocess_text(text):
    """
    Function to clean the input text by:
    - Removing special characters and digits
    - Converting text to lowercase
    - Removing stopwords
    """
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\d+', ' ', text)  # Remove digits
    text = text.lower()  # Convert to lowercase
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Read train data from txt file
def read_train_txt(file_path):
    """
    Function to read training data from a text file.
    Expected format: ID ::: TITLE ::: GENRE ::: DESCRIPTION
    """
    train_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                parts = line.strip().split(" ::: ")
                if len(parts) == 4:  # Ensure it has 4 parts (ID, TITLE, GENRE, DESCRIPTION)
                    train_data.append({
                        'ID': parts[0],
                        'Title': parts[1],
                        'Genre': parts[2],
                        'Description': parts[3]
                    })
    return pd.DataFrame(train_data)

# Read test data from txt file
def read_test_txt(file_path):
    """
    Function to read test data from a text file.
    Expected format: ID ::: TITLE ::: DESCRIPTION
    """
    test_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                parts = line.strip().split(" ::: ")
                if len(parts) == 3:  # Ensure it has 3 parts (ID, TITLE, DESCRIPTION)
                    test_data.append({
                        'ID': parts[0],
                        'Title': parts[1],
                        'Description': parts[2]
                    })
    return pd.DataFrame(test_data)

# Load train and test data
train_data = read_train_txt('train_data.txt')
test_data = read_test_txt('test_data.txt')

# Preprocess the descriptions
train_data['cleaned_description'] = train_data['Description'].apply(preprocess_text)
test_data['cleaned_description'] = test_data['Description'].apply(preprocess_text)

# Split training data into features (X) and target (y)
X_train = train_data['cleaned_description']
y_train = train_data['Genre']

# Vectorize the text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 words for efficiency
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(test_data['cleaned_description'])

# =================== Naive Bayes Classifier ===================

# Initialize the Naive Bayes classifier
naive_bayes_model = MultinomialNB()

# Train the model on the TF-IDF transformed training data
naive_bayes_model.fit(X_train_tfidf, y_train)

# Predict the genres of the test data
test_data['Predicted_Genre_NB'] = naive_bayes_model.predict(X_test_tfidf)

# Print the results
print("----------- Naive Bayes Results -----------")
y_pred_train_nb = naive_bayes_model.predict(X_train_tfidf)
print(f"Naive Bayes Training Accuracy: {accuracy_score(y_train, y_pred_train_nb):.2f}")
print("Classification Report for Naive Bayes:\n", classification_report(y_train, y_pred_train_nb))

# --------------------------------------------------------------
# =================== Logistic Regression Classifier ===================

# Initialize the Logistic Regression classifier
logistic_model = LogisticRegression(max_iter=1000)  # Allow for more iterations to ensure convergence

# Train the model on the TF-IDF transformed training data
logistic_model.fit(X_train_tfidf, y_train)

# Predict the genres of the test data
test_data['Predicted_Genre_LR'] = logistic_model.predict(X_test_tfidf)

# Print the results
print("----------- Logistic Regression Results -----------")
y_pred_train_lr = logistic_model.predict(X_train_tfidf)
print(f"Logistic Regression Training Accuracy: {accuracy_score(y_train, y_pred_train_lr):.2f}")
print("Classification Report for Logistic Regression:\n", classification_report(y_train, y_pred_train_lr))

# --------------------------------------------------------------
# =================== Support Vector Machine Classifier ===================

# Initialize the SVM classifier with a linear kernel (good for text data)
svm_model = SVC(kernel='linear')

# Train the model on the TF-IDF transformed training data
svm_model.fit(X_train_tfidf, y_train)

# Predict the genres of the test data
test_data['Predicted_Genre_SVM'] = svm_model.predict(X_test_tfidf)

# Print the results
print("----------- SVM Results -----------")
y_pred_train_svm = svm_model.predict(X_train_tfidf)
print(f"SVM Training Accuracy: {accuracy_score(y_train, y_pred_train_svm):.2f}")
print("Classification Report for SVM:\n", classification_report(y_train, y_pred_train_svm))

# --------------------------------------------------------------
# Save the predictions to separate text files for each classifier
test_data[['ID', 'Title', 'Predicted_Genre_NB']].to_csv('predicted_genres_naive_bayes.txt', sep=' ::: ', index=False, header=False)
test_data[['ID', 'Title', 'Predicted_Genre_LR']].to_csv('predicted_genres_logistic_regression.txt', sep=' ::: ', index=False, header=False)
test_data[['ID', 'Title', 'Predicted_Genre_SVM']].to_csv('predicted_genres_svm.txt', sep=' ::: ', index=False, header=False)