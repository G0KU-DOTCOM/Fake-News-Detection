import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #representign the articles as numbers
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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
    ('classifier', LinearSVC())
])

param_grid = {
    'vectorizer__max_df': [0.5, 0.7, 0.9],
    'classifier__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

articles_predicted_correctly = int(len(y_test)) * accuracy

print(f"There were {articles_predicted_correctly} articles predicted correctly out of {len(y_test)} articles")

def predict_article(article, model, target_language = 'en'):
    translator = Translator()
    translated_article = translator.translate(article, dest=target_language).text
    vectorized_text = model.named_steps['vectorizer'].transform([translated_article])
    prediction = model.named_steps['classifier'].predict(vectorized_text)
    return "REAL" if prediction == 0 else "FAKE"

test_article = x_test.iloc[10]
prediction = predict_article(test_article, best_model)
print(f"Prediction: {prediction}")


## If you want to add an article:

with open ("mytext.txt", "w", encoding="utf-8") as f:
    f.write(test_article) #you can add any article you want to test

with open ("mytext.txt", "r", encoding="utf-8") as f:
    article = f.read()

custom_prediction = predict_article(test_article, best_model)
print(f"Prediction for the custom article: {custom_prediction}")

