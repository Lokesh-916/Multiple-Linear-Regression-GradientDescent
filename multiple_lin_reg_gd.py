import numpy as np
import pandas as pd

class LinearRegressionGD:
    def __init__(self, alpha=0.01, epochs=300):
        self.alpha = alpha
        self.epochs = epochs
        self.theta = None
        self.x_mean = None
        self.x_std = None

    def preprocess(self, X):
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        X_scaled = (X - self.x_mean) / self.x_std
        return np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

    def fit(self, X, y):
        X = self.preprocess(X)
        n, d = X.shape
        self.theta = np.zeros((d, 1))
        for _ in range(self.epochs):
            gradient = (X.T @ (X @ self.theta - y)) / n
            self.theta -= self.alpha * gradient
        return self

    def predict(self, X):
        X_scaled = (X - self.x_mean) / self.x_std
        X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return X_scaled @ self.theta

    def mse(self, X, y):
        preds = self.predict(X)
        return np.sum((preds - y) ** 2) / (2 * len(y))


class HousingData:
    def __init__(self, data):
        self.data = np.array(data)
        self.df = pd.DataFrame(self.data, columns=['Size', 'Bedrooms', 'Age', 'Price'])

    def describe(self):
        for col in self.df.columns:
            print(f'{self.df[col].mean():.2f} {np.std(self.df[col]):.2f} {self.df[col].min():.2f} {self.df[col].max():.2f}')


if __name__ == "__main__":
    data = [
        [210, 3, 20, 400],
        [160, 2, 15, 330],
        [240, 4, 30, 369],
        [141, 3, 12, 232],
        [300, 4, 8, 540],
        [198, 2, 18, 360]
    ]

    house = HousingData(data)
    X, y = house.data[:, :-1], house.data[:, -1].reshape(-1, 1)

    model = LinearRegressionGD(alpha=0.01, epochs=300)
    model.fit(X, y)

    print("Theta:", [round(v, 3) for v in model.theta.flatten()])
    print("Final MSE:", round(model.mse(X, y), 2))
    house.describe()

    test1 = np.array([[150, 3, 5]])
    test2 = np.array([[200, 4, 2]])
    print("Prediction 1:", round(model.predict(test1)[0][0], 2))
    print("Prediction 2:", round(model.predict(test2)[0][0], 2))
