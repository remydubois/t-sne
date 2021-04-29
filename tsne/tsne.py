import numpy as np
from tsne.barnes_hut_gradients import barnes_hut_gradient_descent
from tsne.conditional import compute_joint_gaussian
from tsne.gradients import exact_gradient_descent
from tsne.pca import project_pca

# import logging

# logger = logging.getLogger('TSNE')


class TSNE:
    def __init__(
        self,
        dimension,
        init="random",
        perplexity=30.0,
        lr=10.0,
        verbose=0,
        min_improvement=1.0e-3,
        nn=-1,
        method="exact",
        theta=0.1,
    ):

        if method not in ["bh", "exact"]:
            raise ValueError("Method should either be `exact` or `bh`")
        if init not in ["pca", "random"]:
            raise ValueError('Init should either be "pca" or "random"')
        self.method = method
        self.theta = theta
        self.dimension = int(dimension)
        self.perplexity = perplexity
        self.lr = lr
        self.verbose = verbose
        self.min_improvement = min_improvement
        self.nn = nn
        self.init = init

    def fit_transform(self, X):
        if self.init == "random":
            Y = np.random.default_rng(0).normal(-1.0, 1.0, (len(X), self.dimension))
        elif self.init == "pca":
            if x.size > 500 ** 2:
                print("[TSNE] Full PCA can be slow for large datasets")
            Y = project_pca(X, 2)
        else:
            raise ValueError

        P = compute_joint_gaussian(
            X, desired_perplexity=self.perplexity, nn=self.nn, verbose=self.verbose
        )
        if self.method == "exact":
            exact_gradient_descent(
                P, Y, min_improvement=self.min_improvement, verbose=self.verbose, lr=self.lr
            )
        elif self.method == "bh":
            barnes_hut_gradient_descent(
                P,
                Y,
                theta=self.theta,
                min_improvement=self.min_improvement,
                verbose=self.verbose,
                lr=self.lr,
            )
        return Y
