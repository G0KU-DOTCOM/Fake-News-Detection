import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #representign the articles as numbers
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from googletrans import Translator

data = pd.read_csv('fake_or_real_news.csv')

data["fake"] = data["label"].apply(lambda x: 0 if x == "REAL" else 1)
data = data.drop("label", axis=1)
#print(data.head())

x, y = data['text'], data['fake']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#len(x_train), len(x_test)

pipeline = Pipeline([
('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LinearSVC())
])

param_grid = {
    'vectorizer_max_df': [0.7, 0.9, 1.0],
    'clf_C': [0.1, 1, 10, 100, 1000]
}

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

clf = LinearSVC()
clf.fit(x_train_vectorized, y_train)

clf.score(x_test_vectorized, y_test)

print(clf.score(x_test_vectorized, y_test))

articles_predicted_correctly = len(y_test) * clf.score(x_test_vectorized, y_test)

print(f"There were {articles_predicted_correctly} articles predicted correctly out of {len(y_test)} articles")



## If you want to add an article:

with open ("mytext.txt", "w", encoding="utf-8") as f:
    f.write(x_test.iloc[10]) #you can add any article you want to test

with open ("mytext.txt", "r", encoding="utf-8") as f:
    article = f.read()

vectorized_text = vectorizer.transform([article])

clf.predict(vectorized_text)

