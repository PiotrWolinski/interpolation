import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt

def seperate(func):
    def inner(*args, **kwargs):
        print('=' * 50)
        print(f'Starting {func.__name__}')
        return func(*args, **kwargs)

    return inner

# Creates banded matrix with dimensions equal to size x size and 
# band consisting of 5 elements spread on 5 diagonals
def create_band_matrix(size: int, band: list[int]=None) -> np.ndarray:
    if band is None or len(band) > 3:
        band = [1, 1, 1]
    
    A = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                A[i][j] = band[0]

            if i == j + 1 or i == j - 1:
                A[i][j] = band[1]

            if i == j + 2 or i == j - 2:
                A[i][j] = band[2]
    
    return A

# Creates vector b with values defined by the expression
# sin(n * (f + 1)) where n is n-th element in the vector
def create_b_vector(size: int, f: int) -> np.ndarray:
    b = np.zeros((size, 1))

    for i in range(size):
        b[i][0] = np.sin(i * (f + 1))

    return b

def pivot(A: np.ndarray) -> np.ndarray:
    pass

@seperate
def solve_lu_factorization(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = A.shape[0]

    print(f'Matrix size = {m}')
    print('Started solving with LU factorization...')
    
    U = A.copy()
    L = np.eye(m)
    x = np.zeros(m)

    U = pivot(U)

    for k in range(m - 1):
        for j in range(k + 1, m):
            L[j][k] = U[j][k] / U[k][k]
            U[j][k:m] = U[j][k:m] - (L[j][k] * U[k][k:m])

    y = forward_substitution(L, b)

    x = backward_substitution(U, y)

    res = np.linalg.norm((A @ x) - b)

    print('Finished solving with LU factorization')
    print(f'Residuum norm = {res}')

    return x

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

@seperate
def plot_times(N: list[int], jacobi: list[float], gauss_seidl: list[float], lu: list[float]):
    plt.plot(N, jacobi, label='Jacobi')
    plt.plot(N, gauss_seidl, label='Gauss-Seidl')
    plt.plot(N, lu, label='LU factorization')

    plt.xlabel('Matrices dimenions [j]')
    plt.ylabel('Time [s]')
    plt.title('Time comparison between Jacobi and Gauss-Seidl methods')
    plt.legend()
    plt.show()