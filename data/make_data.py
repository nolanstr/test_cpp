import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph


models = ["1.0*sin(X_0) + 1.0",
          "1.0*sin(X_0) + X_1 + 1.0",
          "1.0*sin(X_0) + X_1 - 1.0*X_2 + 1.0"]


dims = [1,2,3]
n = 100

for model, dim in zip(models, dims):

    X = np.random.uniform(size=(n,dim))
    y = AGraph(equation=model).evaluate_equation_at(X) + \
                        np.random.normal(loc=0, scale=0.1, size=(n,1))
    data = np.hstack((X, y))
    np.save(f"d{dim}_data", data)
