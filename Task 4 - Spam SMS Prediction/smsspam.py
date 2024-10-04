import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

data = pd.read_csv('spam.csv', encoding='latin-1')

data.drop_duplicates(inplace=True)
data['label'] = data['v1'].map({'ham': 'ham', 'spam': 'spam'})
X = data['v2']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS'])
progress_bar = tqdm(total=100, position=0, leave=True)

# Simulate progress updates
for i in range(10, 101, 10):
    progress_bar.update(10)
    progress_bar.set_description(f'Progress: {i}%')
progress_bar.close()
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
