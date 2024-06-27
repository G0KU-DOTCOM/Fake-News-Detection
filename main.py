import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #representign the articles as numbers
from sklearn.svm import LinearSVC

data = pd.read_csv('fake_or_real_news.csv')

