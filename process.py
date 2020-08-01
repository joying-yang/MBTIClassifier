from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

def get_mbti(text):
    my_lr = load('model.joblib')
    vectorizer = load('scale.joblib')
    data = vectorizer.transform([text])
    return(str(my_lr.predict(data))[2:-2])
