from tnlearn import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tnlearn import VecSymRegressor

X, y = make_classification(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

neuron = VecSymRegressor(max_generations=1)

neuron.fit(X_train, y_train)
print('*' * 20)
print(neuron.neuron)

layers_list = [10, 10, 10]
clf = MLPClassifier()

clf.fit(X_train, y_train)
print(f"Score: {clf.score(X_test, y_test)}")

clf.save(path='my_model_dir', filename='mlp_classifier.pth')
clf.load(path='my_model_dir', filename='mlp_classifier.pth', input_dim=X_train.shape[1], output_dim=1)
clf.fit(X_train, y_train)