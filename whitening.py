import numpy as np
from scipy import linalg


def variance(X, ddof=0.0):
    '''
    Welford algorithm
    http://www.johndcook.com/blog/standard_deviation/
    http://adorio-research.org/wordpress/?p=242
    http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    '''
    print 'Calculating variance and mean...'
    n = 0
    mean = 0.0
    S = 0.0

    for x in X:
        n += 1
        delta = x - mean
        mean += delta/n
        S += delta*(x - mean)  # This expression uses the new value

    return (S/(n - ddof), mean)


def whitening(X):

    '''
    X :         N x d
    Xcont :     N x d
    xZCAWhite : N x d
    W :         d x d
    M :         1 x d
    '''

    # Normalize the brightness and contrast of the patches
    # Compute the mean pixel intensity value separately for each patch.
#    mean_ = np.mean(X, axis=1, dtype='float64', keepdims=True)
#    var_ = np.var(X, axis=1, dtype='float64', ddof=1, keepdims=True)
    var_, mean_ = variance(X.T, ddof=1)
    var_ = var_.reshape(-1,1)
    mean_ = mean_.reshape(-1,1)

    # A small value is added to the variance before division to avoid divide
    # by zero and also suppress noise.
    #
    # Keep in mind that proper choices of the parameters for normalization
    # and whitening can sometimes require adjustment for new data sources.
    #
    # For pixel intensities in the range [0 255], adding 10 to the
    # variance is often a good starting point
    epsilon = 10
    Xcont = (X - mean_) / np.sqrt(var_+epsilon)

    ## Implement ZCA whitening
    #  Now implement ZCA whitening to produce the matrix xZCAWhite.
    #  Visualise the data and compare it to the raw data. You should observe
    #  that whitening results in, among other things, enhanced edges.

    sigma = np.dot(Xcont.T, Xcont) / X.shape[0]
    U, S, V = linalg.svd(sigma)

    # For contrast-normalized data, setting epsilon to 0.01 for 16-by-16 pixel
    # patches, or 0.1 for 8-by-8 pixel patches is a good starting point.
    # Though these are likely best set by cross validation, they can often be
    # tuned visually (e.g., to yield image patches with high contrast, not too
    # much noise, and not too much low-frequency undulation).
    epsilon = 0.1
    W = np.dot(np.dot(U, np.diag(1/np.sqrt(S + epsilon))), U.T)
    M = np.mean(Xcont, axis=0, keepdims=True)
    xZCAWhite = np.dot(Xcont - M, W)

    return Xcont, xZCAWhite, W, M
