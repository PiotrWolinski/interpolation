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
            points_amount = 2
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
    
    def basis(self, j):
        b = [(self.xx - self.X[m]) / (self.X[j] - self.X[m]) for m in range(self.n) if m != j]
        return np.prod(b, axis=0) * self.Y[j]

    def interpolate(self):
        self.xx = np.arange(start=self.X[0], stop=self.X[-1]+0.1, step=0.1)
        b = [self.basis(j) for j in range(self.n)]
        self._values = np.sum(b, axis=0)

    def rmsd(self):

        def find_id(x):
            id = 0
            for i in range(len(self.xx)):
                if self.xx[i] > x:
                    return id
                id += 1
            return id-1

        rmsd = 0.0

        for i in range(len(self.data_X)):
            id = find_id(self.data_X[i])
            mse = (self.values[id] - self.data_Y[i]) ** 2
            rmsd += np.sqrt(mse)

        return rmsd / len(self.data_X)

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
        plt.plot(self.data_X, self.data_Y, label="Actual values")
        plt.plot(self.xx, self.values, color='red', label="Interpolated values")
        ax = plt.gca()
        ax.set_ylim(min(self.data_Y) * 0.9, max(self.data_Y) * 1.1)
        plt.legend()
        plt.xlabel('Odleglosc [m]')
        plt.ylabel('Wysokosc [m]')
        plt.title(f'Interpolacja {self.name} dla {self.n} punktow')
        plt.scatter(self.X, self.Y, color='red')
        plt.show()

        print(f'{self.name} Lagrange RMSD dla {self.points_limit} punktów = {self.rmsd()}')

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
            points_amount = 2
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

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def row(self):
        cur_row = self._row
        self._row += 1
        return cur_row

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def values(self):
        return self._values
    
    def first_condition(self, A: np.ndarray, interval: int, point: int, row: int):
        h = self.X[point] - self.X[interval]

        if h == 0:
            A[row][4*interval] = 1
        else:
            A[row][4*interval] = 1
            A[row][4*interval+1] = h
            A[row][4*interval+2] = h ** 2
            A[row][4*interval+3] = h ** 3
    
    def first_derivative(self, A, interval, row:int):
        # interval = x[i], point = x[i+1]
        h = self.X[interval+1] - self.X[interval]

        A[row][4*interval+1] = 1
        A[row][4*interval+2] = 2 * h
        A[row][4*interval+3] = 3 * (h ** 2)
        A[row][4*interval+5] = -1
    
    def second_derivative(self, A, interval, row:int):
        # interval = x[i], point = x[i+1]
        h = self.X[interval+1] - self.X[interval]

        A[row][4*interval+2] = 2
        A[row][4*interval+3] = 6 * h
        A[row][4*interval+6] = -2

    def set_coeffs(self, coeffs_list: np.ndarray):
        self._coeffs = []

        for i in range(0, len(coeffs_list), 4):
            spline = []
            spline.append(coeffs_list[i])
            spline.append(coeffs_list[i+1])
            spline.append(coeffs_list[i+2])
            spline.append(coeffs_list[i+3])

            self._coeffs.append(spline)
        
    def get_value(self, x: float) -> float:
        spline_id = self.find_spline_id(x)
        spline = self.coeffs[spline_id]

        h = x - self.X[spline_id]

        value = spline[0]
        value += spline[1] * h
        value += spline[2] * (h ** 2)
        value += spline[3] * (h ** 3)

        return value

    def find_spline_id(self, x: float) -> int:
        id = 0

        for i in range(1,len(self.X)):
            if self.X[i] > x:
                return id

            id += 1

        return id-1

    def rmsd(self):
        rmsd = 0.0

        for i in range(len(self.data_X)):
            y = self.get_value(self.data_X[i])
            mse = (y - self.data_Y[i]) ** 2
            rmsd += np.sqrt(mse)

        return rmsd / len(self.data_X)

    def interpolate(self):
        self.intervals_amount = len(self.X) - 1
        
        self._row = 0
        size = 2 * self.intervals_amount + 2 * (self.intervals_amount - 1) + 2
        A = np.zeros((size, size))

        for i in range(self.intervals_amount):
            self.first_condition(A, i, i, self.row)
            self.first_condition(A, i, i+1, self.row)

        for i in range(self.intervals_amount-1):
            self.first_derivative(A, i, self.row)
        
        for i in range(self.intervals_amount-1):
            self.second_derivative(A, i, self.row)

        A[self.row][2] = 1
        A[-1][size-2] = 2
        A[-1][size-1] = 6 * (self.X[-1] - self.X[-2])

        b = []

        for i in range(self.intervals_amount):
            b.append(self.Y[i])
            b.append(self.Y[i+1])

        for i in range(size-2*self.intervals_amount):
            b.append(0)

        b = np.array(b, dtype=np.float32)

        # print(f'A = {A}')

        x, piv = lu_factorization(A)
        # print(f'A = {A}')

        b = b[piv]

        # print(f'b = {b}')
        y = forward_substitution(A, b) # forward_substitution ufsub
        x = backward_substitution(A, y)  # backward_substitution bsub

        # print(f'x = {x}')
        
        self.set_coeffs(x)

        self.xx = np.arange(start=self.X[0], stop=self.X[-1]+0.1, step=0.1) 

        self._values = []

        for x in self.xx:
            self._values.append(self.get_value(x))


    def plot(self):
        plt.plot(self.data_X, self.data_Y, label="Actual values")
        plt.plot(self.xx, self.values, color='red', label="Interpolated values")
        ax = plt.gca()
        ax.set_ylim(min(self.data_Y) * 0.9, max(self.data_Y) * 1.1)
        plt.legend()
        plt.xlabel('Odleglosc [m]')
        plt.ylabel('Wysokosc [m]')
        plt.title(f'Interpolacja {self.name} dla {self.n} punktow')
        plt.scatter(self.X, self.Y, color='red')
        plt.show()

        print(f'{self.name} funkcje sklejane {"losowe" if self.random else ""} RMSD dla {self.points_limit} punktów = {self.rmsd()}')
    
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
    x = b[::]

    for i in range(size): 
        for j in range(i):
            x[i] -= L[i,j]*x[j]
    return x

def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    size = U.shape[0]
    x = y[::]

    for i in range(size-1,-1,-1): 
        for j in range(i+1, size):
            x[i] -= U[i,j]*x[j]
        x[i] = x[i]/U[i,i]
    return x