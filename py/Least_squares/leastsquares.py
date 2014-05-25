# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class leastsquares(object):

    @staticmethod
    def regression(x, y, b=100, sigmaList=np.logspace(-3, 2, 6),
                      lmdList=np.logspace(-3, 2, 6), CV=5):
        """

        :rtype : lambda function for estimating a output from input.
        """
        center_index = np.random.permutation(n)
        center_index = center_index[:b]
        cx = x[center_index, :]
        xx_sum = np.sum(np.power(x, 2), axis=1)
        d2 = np.tile(xx_sum, (1, b)) + np.tile(np.sum(np.power(cx, 2), axis=1).T, (n, 1)) - 2 * x * cx.T

        cv_index = np.floor((np.random.permutation(n) * CV) / n)
        cv_error = np.zeros((sigmaList.size, lmdList.size))
        for sigmaIndex in range(0, sigmaList.size):
            K = np.exp(-d2 / (2 * sigmaList[sigmaIndex] ** 2))
            for cv in range(0, CV):
                Ktr = K[cv_index != cv, :]
                Kte = K[cv_index == cv, :]
                ytr = y[cv_index != cv]
                yte = y[cv_index == cv]
                for lmdIndex in range(0, lmdList.size):
                    cv_alpha = np.linalg.solve(Ktr.T * Ktr + lmdList[lmdIndex] * np.eye(b), Ktr.T * ytr)
                    cv_error[sigmaIndex, lmdIndex] = cv_error[sigmaIndex, lmdIndex] \
                                                     + np.sum(np.power(Kte * cv_alpha - yte, 2), axis=0)

        cv_error_min = np.min(cv_error, axis=1)
        lmdCanList = np.argmin(cv_error, axis=1)
        sigmaCan = np.argmin(cv_error_min, axis=0)
        lmdCan = lmdCanList[sigmaCan]
        sigma = sigmaList[sigmaCan]
        lmd = lmdList[lmdCan]
        X = np.exp(-d2 / (2 * sigma ** 2))
        alpha = np.linalg.solve(X.T * X + lmd * np.eye(b), X.T * y)

        f = (lambda x:
             np.exp(-(np.tile(np.sum(np.power(x, 2), axis=1), (1, b))
                      + np.tile(np.sum(np.power(cx, 2), axis=1).T, (x.shape[1], 1)) - 2 * x * cx.T)
                    / (2 * sigma ** 2)) * alpha)
        return f


if __name__ == "__main__":
    # x : n by d matrix
    n = 100
    noiseLevel = .1
    x = np.mat(np.linspace(0, 1, n)).T
    y_true = np.sin(2 * np.pi * x)
    y = y_true + noiseLevel * np.random.randn(n, 1)

    nt = 200
    xt = np.mat(np.linspace(0, 1, nt)).T
    f = leastsquares.regression(x, y)
    yh = f(xt)

    fig = plt.figure()
    fig.clear()
    fig.hold()
    plt.plot(np.array(x), np.array(y), 'rx')
    plt.plot(np.array(x), np.array(y_true), 'r')
    plt.plot(np.array(xt), np.array(yh), 'b')
    plt.show()

#TODO: write sklearn version of least-square method.

# <codecell>


