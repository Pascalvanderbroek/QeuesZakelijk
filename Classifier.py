import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_matrix(data):
    plt.figure()
    plt.matshow(data)
    plt.show()


def conf_els(conf, labels):
    tot = conf.sum().sum()
    cols = conf.sum(axis=0)
    rows = conf.sum(axis=1)

    tp = conf.diagonal()
    fp = cols - tp
    fn = rows - tp
    tn = tot - tp - fp - fn

    res = list(zip(labels, tp, fp, fn, tn))
    return res


def conf_data(metrics):
    tp = sum([foo[1] for foo in metrics])
    fp = sum([foo[2] for foo in metrics])
    fn = sum([foo[3] for foo in metrics])
    tn = sum([foo[4] for foo in metrics])

    rv = {}
    rv['tpr'] = tp / (tp + fn)
    rv['ppv'] = tp / (tp + fp)
    rv['tnr'] = tn / (tn + fp)
    rv['fpr'] = fp / (fp + tn)

    return rv


data = pd.read_csv("Zakelijk_SpeechInput_Orgineel.csv", encoding='ISO-8859-1')
data = data.drop(data.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], axis=1)
data = data.fillna('-')

pipeline = Pipeline([('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)),
                     ('clf', SVC())])

X_train, X_test, y_train, y_test = train_test_split(data['User Input'], data['Label'], test_size=0.2, random_state=1)

scores = cross_validate(pipeline, X_train, y_train)
print(scores)
scores['test_score'].mean()

clfs = []
clfs.append(LogisticRegression())
clfs.append(SVC(kernel='linear'))
clfs.append(KNeighborsClassifier(n_neighbors=5))
clfs.append(DecisionTreeClassifier())
clfs.append(RandomForestClassifier())

for classifier in clfs:
    pipeline.set_params(clf=classifier)
    scores = cross_validate(pipeline, X_train, y_train)
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for key, values in scores.items():
        print(key, ' mean ', values.mean())
        print(key, ' std ', values.std())

pipeline.set_params(clf=SVC())
pipeline.steps
cv_grid = GridSearchCV(pipeline, param_grid={
    'clf__kernel': ['linear', 'rbf'],
    'clf__C': np.linspace(0.1, 1.2, 12)
})

cv_grid.fit(X_train, y_train)

cv_grid.best_params_

cv_grid.best_estimator_
cv_grid.best_score_

y_predict = cv_grid.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print('-------------------')
print('Accuracy of the best classifier after CV is %.3f%%' % (accuracy * 100))
print('-------------------')
print('Classification Report ->')
print(classification_report(y_predict, y_test))
print('----------------------------------------')
print('Confusion Matrix ->')
print(confusion_matrix(y_predict, y_test))
data = np.array(confusion_matrix(y_predict, y_test))
plot_matrix(data)

labels = ['Complex','DBZ1','DBZ2','Simpel','Technisch']

TP = np.diagonal(data)
FP = np.sum(data, axis=0) - TP
FN = np.sum(data, axis=1) - TP
TN = data.sum() - TP - FP - FN

conf_els(data, labels)

metrics = conf_els(data, labels)
print(metrics)
print("Bepalen van de scores:")
scores = conf_data(metrics)
print(scores)
