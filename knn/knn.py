import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class CalculateDistances:
    def euclidian(self, a, b):
        """
        sqrt((a-b)^2)

        :param a: first variable of the equation
        :param b: second variable of the equation
        :return: euclidian distance between `a` and `b`
        """
        return np.sqrt(np.sum(np.power(a - b, 2), axis=1))

    def manhattan(self, a, b):
        """
        |a - b|

        :param a: first variable of the equation
        :param b: second variable of the equation
        :return: manhattan distance between `a` and `b`
        """
        return np.sum(np.abs(a - b), axis=1)

class KNN:
    def __init__(self, k, distance='euclidian'):
        """
        :param k: number of neighbors
        :param distance: metric used to measure the distance between neighbors
        """
        assert distance in ['euclidian', 'manhattan'], 'Distance calculation available: [euclidian, manhattan]'
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        """
        Fits the data to the model

        :param X: input values
        :param y: classes
        """
        self.X = self._normalize(X)
        self.y = y

    def predict(self, X_test):
        """
        Predictions for input values

        :param X_test: input values
        :return y_pred: class prediction for input values
        """
        distances = self._calculate_distance(X_test)

        if distances.ndim == 1:
            distances = distances.reshape(1, distances.shape[0])
            num_preds = 1
        else:
            num_preds = distances.shape[0]

        y_pred = np.empty((num_preds), dtype=np.int32)

        for i in range(num_preds):
            indexes = np.argsort(distances[i, :])
            classes = self.y[indexes[:self.k]]
            values, distribution = np.unique(classes, return_counts=True)
            y_pred[i] = values[np.argmax(distribution)]

        return y_pred

    def _normalize(self, target):
        """
        Normalizes input values ​​according to Min-Max normalization [x-min(x) / max(x)-min(x)]

        :param target: input values
        :return: normalized data
        """
        return (target - target.min()) / (target.max() - target.min())

    def _calculate_distance(self, target):
        """
        Calculates the distance from neighbors to input values ​​according to the established calculation metric

        :param target: input values
        :return distances: distances from neighbors to input values
        """
        target = self._normalize(target)
        distance_method = getattr(CalculateDistances, self.distance)
        distances = np.empty((target.shape[0], self.y.shape[0]), dtype=np.float64)

        try:
            for i in range(target.shape[0]):
                distances[i, :] = distance_method(self, self.X, target[i, :]) 

        except IndexError:
            distances = distance_method(self, self.X, target)

        return np.array(distances)

    def plot_decision_boundaries(self, h=0.02, show_actual_data=True):
        '''
        Shows the decision boundaries

        :return:
        :reference: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
        '''

        _, ax = plt.subplots()
        
        # 2 classes -> 2 colors
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, cmap=cmap_light)
        if show_actual_data:
            ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=cmap_bold, edgecolor='k')

        # data is normalized [0, 1]
        ax.set_xlim([-.25, 1.25])
        ax.set_ylim([-.25, 1.25])

        return ax
        

class_1 = np.loadtxt("data/class_1.txt", delimiter=";")
class_2 = np.loadtxt("data/class_2.txt", delimiter=";")
dataset = np.vstack([class_1, class_2])

X = dataset[:, [0, 1]]
y = dataset[:, 2]

knn = KNN(k=3)
knn.fit(X, y)

X_test = np.random.randint(10, size=(20, 2), )
preds = knn.predict(X_test)
print(preds)

ax = knn.plot_decision_boundaries(show_actual_data=True)
norm_test = knn._normalize(X_test)
plt.scatter(norm_test[:, 0], norm_test[:, 1], marker='x', c='white')
plt.show()
