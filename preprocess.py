from kmeans import reg_kmeans
from whitening import whitening
from encoder import encoder, encoder_slow
from utils import draw_patches
import numpy as np
import constants
import h5py
import operator as op


def create_label_selector(operator, val):
    def label_selector(label):
        return operator(label, val)
    return label_selector


def extract_patches(images, labels, sel):

    np.random.seed(999)
    imSize = constants.imSize
    patchSize = constants.patchSize
    nPatches = constants.nPatches
    nChannels = constants.nChannels
    
    top = np.random.random_integers(0, imSize-patchSize, nPatches)
    left = np.random.random_integers(0, imSize-patchSize, nPatches)
    d = np.where(sel(np.asarray(labels)))[0]

    patches = np.vstack([images[d[i%len(d)]].reshape((nChannels,imSize,imSize))\
                            [:, y:y+patchSize, x:x+patchSize].reshape((1,-1))
                            for i, (x, y) in enumerate(zip(top, left))])
    return patches


def preprocess(*args):
    patches = extract_patches(*args)
    Xcont, xZCAWhite, W, M = whitening(patches)
    nCentroids = constants.nCentroids
    centroids = reg_kmeans(xZCAWhite, k=nCentroids,
                           num_iterations=50,
                           pct_exit=0.01,
                           batch_size=10000, damped=True)
    return centroids, W, M, patches, Xcont, xZCAWhite


if __name__ == '__main__':

    verbose = True
    patchSize = constants.patchSize
    
    conditions = ((op.eq, 0), # ==0 [0]
                  (op.eq, 1), # ==1 [1]
                  (op.eq, 2), # ==2 [2]
                  (op.eq, 3), # ==3 [3]
                  (op.eq, 4), # ==4 [4]
                  (op.ge, 0), # >=0 [5]
                  (op.ge, 1), # >=1 [6]
                  (op.ge, 2), # >=2 [7]
                  (op.ge, 3), # >=3 [8]
                  (op.le, 1), # <=1 [9]
                  (op.le, 2), # <=2 [10]
                  (op.le, 3)) # <=3 [11]
    sels = [create_label_selector(*p) for p in conditions]

    with h5py.File(constants.train_packed_file, "r") as f:
        images = f.get("X")
        labels = f.get("Y")
        centroids, W, M, patches, Xcont, xZCAWhite =\
                                            preprocess(images, labels, sels[5])

    if verbose:
        draw_patches(patches[:36], H=patchSize, W=patchSize)
        draw_patches(Xcont[:36], H=patchSize, W=patchSize)
        draw_patches(xZCAWhite[:36], H=patchSize, W=patchSize)
        draw_patches(centroids[:600], H=patchSize, W=patchSize, br=1.0, cont=2)
    del Xcont, xZCAWhite, patches

    with h5py.File(constants.train_packed_file, 'r') as fi:
        images = fi.get('X')
        labels = fi.get('y')
        X = images
        y = labels
        with h5py.File(constants.train_features_file, 'w') as fo:
            XC1 = encoder(X, centroids, W, M, chunk_size=constants.chunk_size,
                          batch_size=constants.batch_size, Y=y, fo=fo)

    with h5py.File(constants.test_packed_file, 'r') as fi:
        images = fi.get('X')
        X = images
        with h5py.File(constants.test_features_file, 'w') as fo:
            XC1 = encoder(X, centroids, W, M, chunk_size=constants.chunk_size,
                          batch_size=constants.batch_size, Y=None, fo=fo)
#            XC2 = encoder_slow(X, centroids, W, M, batch_size=3)
#    np.testing.assert_allclose(XC1, XC2, rtol=1e-2)
