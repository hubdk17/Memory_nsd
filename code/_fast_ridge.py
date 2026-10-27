"""Ridge regression where every alpha comes from a single decomposition.

Uses the dual (n x n) eigendecomposition when n_samples < n_features, which is the
regime here (761 trials vs up to 5,828 voxels). That is orders of magnitude cheaper
than an SVD of the full design matrix and numerically equivalent.
"""
import numpy as np


class SVDRidge:
    """Centred multi-output ridge; call predict(X, alpha) for any alpha after one fit."""

    def fit(self, X, Y):
        self.xm_ = X.mean(0)
        self.ym_ = Y.mean(0)
        Xc = X - self.xm_
        Yc = Y - self.ym_
        n, p = Xc.shape
        self.dual_ = n <= p
        if self.dual_:
            K = Xc @ Xc.T                       # (n, n)
            lam, Q = np.linalg.eigh(K)
            np.maximum(lam, 0.0, out=lam)
            self.lam_ = lam
            self.XtQ_ = Xc.T @ Q                # (p, n)
            self.QtY_ = Q.T @ Yc                # (n, k)
        else:
            G = Xc.T @ Xc                       # (p, p)
            lam, Q = np.linalg.eigh(G)
            np.maximum(lam, 0.0, out=lam)
            self.lam_ = lam
            self.Q_ = Q
            self.QtXtY_ = Q.T @ (Xc.T @ Yc)
        return self

    def coef(self, alpha):
        if self.dual_:
            return (self.XtQ_ / (self.lam_ + alpha)) @ self.QtY_
        return self.Q_ @ (self.QtXtY_ / (self.lam_ + alpha)[:, None])

    def predict(self, X, alpha):
        return (X - self.xm_) @ self.coef(alpha) + self.ym_
