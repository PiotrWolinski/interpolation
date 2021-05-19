import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
import pandas as pd
import random

class LagrangeInterpolation:

    def __init__(self, df: pd.DataFrame, points_limit=12, random=False, filename=None):
        self.name = "" if filename is None else filename.split('.')[0]
        self.data_X = df['Dystans (m)']
        self.data_Y = df['Wysokość (m)']
        self.points_limit = points_limit
        self.random = random
        self.choose_points()
        self.n = len(self.X)

    def choose_points(self):
        self._X = []
        self._Y = []
        data_size = self.data_X.shape[0]

        if self.random:
            # random spacing
            points_amount = 0
            points_tuples = []

            points_tuples.append((self.data_X[0], self.data_Y[0]))
            points_tuples.append((self.data_X[self.data_X.shape[0]-1], self.data_Y[self.data_X.shape[0]-1]))

            while points_amount < self.points_limit:
                i = random.randint(0,data_size - 1)

                if self.data_X[i] not in self._X:
                    points_tuples.append((self.data_X[i], self.data_Y[i]))
                    points_amount += 1

            points_tuples.sort(key=lambda x: x[0])
            for x, y in points_tuples:
                self._X.append(x)
                self._Y.append(y)

        else:
            # even spacing
            step = data_size // (self.points_limit - 1)

            for i in range(0, data_size, step):
                self._X.append(self.data_X[i])
                self._Y.append(self.data_Y[i])

        print(self.X)
        print(self.Y)
    
    def basis(self, j):
        b = [(self.xx - self.X[m]) / (self.X[j] - self.X[m]) for m in range(self.n) if m != j]
        return np.prod(b, axis=0) * self.Y[j]

    def interpolate(self):
        self.xx = np.arange(self.X[0] * 10, (self.X[-1] * 10)) / 10
        b = [self.basis(j) for j in range(self.n)]
        self._values = np.sum(b, axis=0)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def values(self):
        return self._values

    def plot(self):

        # Do plotowania:
        # wszystkie punkty jakie mam razem z ich wartościami
        # interpolowane punkty (tyle ile ich miało być) razem z wartościami
        print(f'Name = {self.name}')
        plt.plot(self.data_X, self.data_Y, label="Actual values")
        print(len(self.xx))
        print(self.values)
        plt.plot(self.xx, self.values, color='red', label="Interpolated values")
        ax = plt.gca()
        ax.set_ylim(min(self.data_Y) * 0.9, max(self.data_Y) * 1.1)
        plt.legend()
        plt.xlabel('Odleglosc [m]')
        plt.ylabel('Wysokosc [m]')
        plt.title(f'Interpolacja dla {self.n} punktow')
        plt.scatter(self.X, self.Y, color='red')
        plt.show()

class SplineInterpolation:

    def __init__(self, df: pd.DataFrame, points_limit=12, random=False, filename=None):
        self.name = "" if filename is None else filename.split('.')[0]
        self.data_X = df['Dystans (m)']
        self.data_Y = df['Wysokość (m)']
        self.points_limit = points_limit
        self.random = random
        self.choose_points()
        self.n = len(self.X)

    def choose_points(self):
        self._X = []
        self._Y = []
        data_size = self.data_X.shape[0]

        if self.random:
            # random spacing
            points_amount = 0
            points_tuples = []

            points_tuples.append((self.data_X[0], self.data_Y[0]))
            points_tuples.append((self.data_X[self.data_X.shape[0]-1], self.data_Y[self.data_X.shape[0]-1]))
            
            while points_amount < self.points_limit:
                i = random.randint(0,data_size - 1)

                if self.data_X[i] not in self._X:
                    points_tuples.append((self.data_X[i], self.data_Y[i]))
                    points_amount += 1

            points_tuples.sort(key=lambda x: x[0])
            for x, y in points_tuples:
                self._X.append(x)
                self._Y.append(y)



def seperate(func):
    def inner(*args, **kwargs):
        print('=' * 50)
        print(f'Starting {func.__name__}')
        return func(*args, **kwargs)

    return inner

def lu_factorization(A: np.ndarray):
    n = A.shape[0]
    piv = np.arange(0,n)
    for k in range(n-1):

        # pivoting
        max_row_index = np.argmax(abs(A[k:n,k])) + k
        piv[[k,max_row_index]] = piv[[max_row_index,k]]
        A[[k,max_row_index]] = A[[max_row_index,k]]

        # LU 
        for i in range(k+1,n):          
            A[i,k] = A[i,k]/A[k,k]      
            for j in range(k+1,n):      
                A[i,j] -= A[i,k]*A[k,j] 

    return [A, piv]

def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    size = L.shape[0]
    x = np.zeros((size, 1))

    for m in range(size):
        if m == 0:
            x[m] = b[m] / L[m][m]
            continue
        
        sub = 0

        for i in range(m):
            sub += L[m][i] * x[i]

        x[m] = (b[m] - sub) / L[m][m]

    return x

def backward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    size = U.shape[0]
    x = np.zeros((size, 1))

    for m in range(size - 1, -1, -1):
        if m == size - 1:
            x[m] = b[m] / U[m][m]
            continue
        
        sub = 0

        for i in range(size - 1, m, -1):
            sub += U[m][i] * x[i]

        x[m] = (b[m] - sub) / U[m][m]

    return x