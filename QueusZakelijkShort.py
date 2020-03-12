import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data = pd.read_csv("Zakelijk_SpeechInput_Orgineel1.csv", encoding='ISO-8859-1')
data = data.drop(data.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], axis=1)
data = data.fillna('-')

X_train, X_test, y_train, y_test = train_test_split(data['User Input'], data['Label'], test_size=0.2, random_state=1)

pipeline = Pipeline([('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)),
                     ('clf', SVC(C=1, kernel='linear', gamma=1))])
