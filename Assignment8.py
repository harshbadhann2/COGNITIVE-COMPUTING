from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris_dataset = datasets.load_iris()
features = iris_dataset.data
labels = iris_dataset.target

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

normalizer = StandardScaler()
features_train = normalizer.fit_transform(features_train)
features_test = normalizer.transform(features_test)

classifier = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=200)
classifier.fit(features_train, labels_train)

predictions = classifier.predict(features_test)

model_accuracy = accuracy_score(labels_test, predictions)
print(f"Accuracy: {model_accuracy:.2f}")