
from numpy.lib.function_base import gradient
import psutil
import  ray
import time
from ray.worker import get
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus, include_dashboard=False)

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

@ray.remote
def get_accuracy(y_test, predictions) -> int:
    distinct_elements = set(y_test).intersection(predictions)
    return (len(predictions) - len(distinct_elements)) / len(y_test)

def euclidean_distance(a,b) -> float:
    return distance.euclidean(a,b)

def fit(X_train, y_train) -> tuple:
    return X_train, y_train

@ray.remote
def predict(X_test):
    predictions = []
    for row in X_test:
        label = closest(row)
        predictions.append(label)
    return predictions

def closest(row):
    best_distance = euclidean_distance(row, X_train[0])
    best_index = 0
    for i in range(1 ,len(X_train)):
        distance = euclidean_distance(row, X_train[i])
        if distance  < best_distance:
            best_distance = distance
            best_index = i
    return y_train[best_index]

start = time.time()

predictions = predict.remote(X_test)
end = time.time()

print("predict time taken ", (end - start))

start = time.time()
accuracy = get_accuracy.remote(y_test, predictions)
end = time.time()
print("accuracy calc time taken ", (end - start))

print("Knn accuracy score: ", ray.get(accuracy))

