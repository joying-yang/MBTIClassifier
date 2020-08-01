import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from bs4 import BeautifulSoup
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    return text


train = pd.read_csv('mbti_1.csv')
print(train)

mbti = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition',
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling',
        'J':'Judging', 'P': 'Perceiving'}

scoring = {'acc': 'accuracy',
           'neg_log_loss': 'neg_log_loss',
           'f1_micro': 'f1_micro'}

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

print(train.shape)

train['clean_posts'] = train['posts'].apply(cleanText)

vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english',
                                                 lowercase = True, max_features = 5000)

X_scaled = vectorizer.fit_transform(train['clean_posts'])
y = train['type']

my_lr = LogisticRegression(random_state=0, max_iter=1000)

param_grid = {'C': np.logspace(-4,4,10)}

grid = GridSearchCV(my_lr, param_grid=param_grid, cv=kfolds, scoring='f1_micro',verbose=1,n_jobs=-1)

model = grid.fit(X_scaled, y)

dump(vectorizer, 'scale.joblib')
dump(model, 'model.joblib')
# results_lr = cross_validate(model_lr, train['clean_posts'], train['type'], cv=kfolds,
#                           scoring=scoring, n_jobs=-1)
#
# print("CV F1: {:0.4f} (+/- {:0.4f})".format(np.mean(results_lr['test_f1_micro']),
#                                                           np.std(results_lr['test_f1_micro'])))
