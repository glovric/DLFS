import numpy as np

def make_linear_regression(n_samples=200, n_features=1, noise=20):

    coef = np.random.uniform(-50, 50, size=n_features)
    bias = np.random.uniform(-10, 10)

    if n_features == 1:
        a = coef[0]
        f = lambda x: a*x + bias
        X = np.random.uniform(-5, 5, size=(n_samples, ))
        y = f(X) + np.random.randn(n_samples) * noise
        return X, y, f
    elif n_features == 2:
        a1, a2 = coef
        f = lambda x1, x2: a1*x1 + a2*x2 + bias
        X = np.random.uniform(-5, 5, size=(n_samples, 2))
        x1, x2 = X[:, 0], X[:, 1]
        y = f(x1, x2) + np.random.randn(n_samples) * noise
        return X, y, f

def make_sine(n_samples=100, n_features=1, noise=0.3):
    if n_features == 2:
        X = np.random.uniform(-5, 5, size=(n_samples, 2))
        x1, x2 = X[:, 0], X[:, 1]
        f = lambda x1, x2: np.sin(x1) * np.cos(x2)
        y = f(x1, x2) + np.random.randn(len(X)) * noise
        return X, y, f        
    else:
        X = np.random.uniform(-5, 5, size=(n_samples, ))
        f = lambda x: 2*np.sin(x)
        y = f(X) + np.random.randn(n_samples) * noise
        return X, y, f

def make_cubic(n_samples=50, n_features=1, noise=5):
    if n_features == 2:
        X = np.random.uniform(-5, 5, size=(n_samples, 2))
        x1, x2 = X[:, 0], X[:, 1]
        f = lambda x1, x2: (x1**3 - 2*x1**2 + 3*x1) * np.cos(x2) + 0.5*x2**2
        y = f(x1, x2) + np.random.randn(len(X)) * noise
        return X, y, f
    else:
        f = lambda x: X**3-2*X**2+3*x+1
        X = np.random.uniform(-5, 5, size=(n_samples, ))
        y = f(X) + np.random.randn(n_samples) * noise
        return X, y, None

def make_logarithm(n_samples=100, n_features=1, noise=0.1):
    if n_features == 2:
        X = np.empty((0, 2))

        while len(X) < n_samples:
            candidates = np.random.uniform(-1, 10, size=(n_samples, 2))
            valid = candidates[:, 0] * candidates[:, 1] + 1 > 0
            X = np.vstack([X, candidates[valid]])

        X = X[:n_samples]
        x1, x2 = X[:, 0], X[:, 1]
        f = lambda x1, x2: np.log(x1 * x2 + 1)
        y = f(x1, x2) + np.random.randn(n_samples) * noise
        return X, y, f
    else:
        X = np.random.uniform(-1, 10, size=(n_samples, ))
        f = lambda x: np.log(x + 1)
        y = f(X) + np.random.randn(n_samples) * noise
        return X, y, f