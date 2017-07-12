import numpy as np
import cPickle
from sklearn.model_selection import train_test_split


with open('pre_train', 'rb') as f:
	dic = cPickle.load(f)
	X = dic['x_']
	X = np.array(X)
	X = X.reshape(X.shape[0], X.shape[2])
	y = dic['y_']
	print 'data loaded.'

X_train, X_ts, y_train, y_ts = train_test_split(X, y, test_size=0.76, stratify=y)
_, X_test, _, y_test = train_test_split(X_ts, y_ts, test_size=0.035, stratify=y_ts)

print 'Train X:', X_train.shape
print 'Test X:', X_test.shape
print 'Train y:', len(y_train)
print 'Test y:', len(y_test)

'''SVC'''
from sklearn import svm
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)

print('SVC...')
print(svc.score(X_test, y_test))
print(rbf_svc.score(X_test, y_test))
print(poly_svc.score(X_test, y_test))
print(lin_svc.score(X_test, y_test))

ovo_svc = svm.SVC(decision_function_shape='ovo').fit(X_train, y_train)
ovr_svc = svm.SVC(decision_function_shape='ovr').fit(X_train, y_train)
print(ovo_svc.score(X_test, y_test))
print(ovr_svc.score(X_test, y_test))

'''Gradient Tree Boosting'''
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

print('Gradient Tree Boosting...')
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                    max_depth=1, random_state=0).fit(X_train, y_train)

print(clf.score(X_test, y_test))

'''Ensemble methods'''
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

print('Decision Tree...')
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0).fit(X_train, y_train)
print(clf.score(X_test, y_test))

print('Random Forest...')
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0).fit(X_train, y_train)
print(clf.score(X_test, y_test))

print('Extremely randomized trees...')
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0).fit(X_train, y_train)
print(clf.score(X_test, y_test))

'''SGD'''
from sklearn.linear_model import SGDClassifier
print('SGD...')
clf = SGDClassifier(loss="log", penalty="l1", alpha=0.001, n_iter=100).fit(X_train, y_train)
print(clf.score(X_test, y_test))


'''Nearest Neighbors'''
from sklearn.neighbors.nearest_centroid import NearestCentroid
print('Nearest Centroid...')
clf = NearestCentroid().fit(X_train, y_train)
print(clf.score(X_test, y_test))


from sklearn import neighbors
print('Nearest Neighbors...')
n_neighbors = 10
for weights in ['uniform', 'distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights).fit(X_train, y_train)
    print(clf.score(X_test, y_test))

'''Neural Network'''
from sklearn.neural_network import MLPClassifier
print('MLPClassifier...')
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))
