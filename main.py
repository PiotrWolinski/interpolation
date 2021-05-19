import functions as func
import pandas as pd

class Solution:

    def __init__(self, path='MountEverest.csv'):
        self._path = path

        self.load_file()

    def load_file(self, path=None):
        if path is None:
            path = self._path

        self._data = pd.read_csv(path)

    def print_file(self):
        print(self._data)

    def lagrange(self):
        lagrange = func.LagrangeInterpolation(self._data, 6, random=True, filename=self._path)
        lagrange.interpolate()
        lagrange.plot()

def main():
    solution = Solution() # 'test.csv'
    solution.lagrange()

if __name__ == '__main__':
    main()