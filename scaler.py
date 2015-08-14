import h5py
import numpy as np


def scale4svm(fin, fout, sd, mean, batch_size=5000):
    with h5py.File(fin, 'r') as fi:
        with h5py.File(fout, "w") as fo:
            if 'y' in fi.keys():
                y = fi.get("y")
                fo.create_dataset("y", data=np.array(y).reshape(-1, 1))

            X = fi.get("X")
            N, d = X.shape
            X_scaled = fo.create_dataset("X", (N, d+1), dtype=float, chunks=True)
            print 'Chunks:', X_scaled.chunks
            for b in xrange(0, N, batch_size):
                print 'Scaling batch %d/%d' % (b/batch_size + 1, np.ceil(1.0*N/batch_size))
                r = slice(b, min(b+batch_size, N))
                X_scaled[r, :-1] = (X[r]-mean)/sd
                X_scaled[r, -1] = 1


def scale(fin, fout, sd, mean, batch_size=5000):
    with h5py.File(fin, 'r') as fi:
        with h5py.File(fout, 'w') as fo:
            if 'y' in fi.keys():
                y = fi.get('y')[()]
                fo.create_dataset('y', data=y)
                print 'y data set has been created'

            X = fi.get('X')
            N, d = X.shape
            X_scaled = fo.create_dataset('X', X.shape, dtype=X.dtype)
            for b in xrange(0, N, batch_size):
                print 'Scaling batch %d/%d' % (b/batch_size + 1, np.ceil(1.0*N/batch_size))
                r = slice(b, min(b+batch_size, N))
                X_scaled[r] = (X[r]-mean)/sd
                

def variance(fin, ddof=0.0):
    '''
    Welford algorithm
    http://www.johndcook.com/blog/standard_deviation/
    http://adorio-research.org/wordpress/?p=242
    http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    '''
    print 'Calculating variance and mean...'
    with h5py.File(fin, 'r') as fi:
        X = fi.get('X')
        n = 0
        mean = 0.0
        S = 0.0
        for x in X:
            n += 1
            delta = x - mean
            mean += delta/n
            S += delta*(x - mean)  # This expression uses the new value
    return (S/(n - ddof), mean)



def fit2distribution(labels, levels):
    fitted = np.zeros(len(levels), dtype=int)
    L = np.sort(np.unique(labels))
    D = 1.0*np.histogram(labels, bins=len(L))[0] * len(levels) / len(labels)
    D = np.intc(np.ceil(D))
    if sum(D) != len(levels):
        D[np.argmax(D)] -= sum(D) - len(levels)
    assert(sum(D) == len(levels))
    idx = np.argsort(levels.flatten())
    b = 0
    for i,d in enumerate(D):
        fitted[idx[b:b+d]] = L[i]
        b += d
    return fitted