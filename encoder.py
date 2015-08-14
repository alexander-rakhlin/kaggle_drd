import theano.tensor as T
from theano.tensor.nnet import conv
from theano import function, config, shared
import theano.sandbox.neighbours as TSN
import numpy as np
import scipy.io
import time

import os
import h5py
import constants


def emergencydump(X, k):
    out = os.path.join(constants.emergency_dir, 'emrg.'+format(k,'09d')+'.hd5')
    with h5py.File(out, 'w') as fo:    
        fo.create_dataset('X', data=X)
        
    
def encoder(image, centroids, W, M, fo=None,
            chunk_size=999, batch_size=333, Y=None):

    '''
    input:
    image :     N x D   (N, n_channels*HSize*WSize)
    centroids : K x d   (K, d=n_channels*psize*psize)
    W :         d x d
    M :         1 x d

    output:
    features:   N x 4*K
    '''

    N = image.shape[0]

    # chunk to allocate on GPU
    n_chunks = np.int(np.ceil(1.0*N/chunk_size))
    # batch to process in a single pass
    n_batches = chunk_size/batch_size
    assert(n_batches*batch_size == chunk_size)

    n_channels = 3
    imSize = np.int(np.sqrt(image.shape[1]/n_channels))
    assert(np.square(imSize)*3 == image.shape[1])

#    image = image.reshape(-1, n_channels, imSize, imSize)
#
#    #pad with zeros
#    pad = np.zeros((chunk_size*n_chunks-N, image.shape[1], image.shape[2],
#                    image.shape[3]), dtype=image.dtype)
#    image = np.vstack([image, pad])

    num_centroids = centroids.shape[0]
    psize = np.int(np.sqrt(centroids.shape[1]/n_channels))
    assert(np.square(psize)*n_channels == centroids.shape[1])
    h = imSize - psize + 1
    w = imSize - psize + 1

    if fo is not None:
        X_dataset = fo.create_dataset("X", (N, 4*num_centroids),
                                     dtype=config.floatX)
        if Y is not None:
            fo.create_dataset("y", data=Y)

    img = np.float32(image[:chunk_size])
    #pad with zeros
    if img.shape[0]<chunk_size:
        pad = np.zeros((chunk_size-img.shape[0], img.shape[1]), dtype=img.dtype)
        img = np.vstack([img, pad])    
    X = shared(img.reshape(-1, n_channels, imSize, imSize), borrow=True)
    C = shared(np.float32(centroids), borrow=True)
    W = shared(np.float32(W), borrow=True)
    M = shared(np.float32(M), borrow=True, broadcastable=(True, False))

    cc = T.square(C).sum(axis=1, keepdims=True).T  # 1 x K

    im = T.tensor4(dtype=config.floatX)

    eyef = T.eye(psize * psize, psize * psize, dtype=config.floatX)[::-1]
    filts = T.reshape(eyef, (psize * psize, psize, psize))
    filts = T.shape_padleft(filts).dimshuffle((1, 0, 2, 3))

    res = T.zeros((n_channels, batch_size, psize * psize, h, w), dtype=config.floatX)

    for i in xrange(n_channels):
        cur_slice = T.shape_padleft(im[:, i, :, :]).dimshuffle((1, 0, 2, 3))
        res = T.set_subtensor(res[i], conv.conv2d(cur_slice, filts))

    # res ~ (channel, batch, conv, hi, wi) -> (batch, hi, wi, channel, conv)
    # -> (batch, hi*wi, channel*h*w)
    res = res.dimshuffle((1, 3, 4, 0, 2)).\
        reshape((batch_size*h*w, n_channels*psize*psize))

    # Normalize the brightness and contrast separately for each patch.
    epsilon = 10
    mean_ = T.cast(res.mean(axis=1, keepdims=True), config.floatX)
    dof = n_channels*psize*psize  # adjust DOF
    var_ = T.cast(res.var(axis=1, keepdims=True), config.floatX)*dof/(dof-1)
    res = (res-mean_)/T.sqrt(var_+epsilon)

    # Whitening
    res = T.dot(res-M, W)                          # batch*h*w x n_channels*psize*psize

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # normalise to unit length
    #res /= T.sqrt(T.sqr(res).sum(axis=1, keepdims=True))

    cx = T.dot(res, C.T)                           # batch*h*w x K
    res = T.sqr(res).sum(axis=1, keepdims=True)    # batch*h*w x 1
    distance = cc - 2*cx + res                     # batch*h*w x K
    distance *= (distance > 0)                     # precision issue
    distance = T.sqrt(distance)
    batch = distance.mean(axis=1, keepdims=True) - distance
    batch = batch.reshape((batch_size, h, w, num_centroids)).\
        dimshuffle(0, 3, 1, 2)                     # batch x K x h x w
    batch *= (batch > 0)    # ReLU

    if np.int(h/2)*2 == h:
        half = np.int(h/2)
        padded = batch
    else:
        half = np.int((h+1)/2)
        padded = T.zeros((batch_size, num_centroids, h+1, w+1))
        padded = T.set_subtensor(padded[:, :, :h, :w], batch)

    pool_sum = TSN.images2neibs(padded, (half, half))   # batch*K*4 x h*w/4
    pool_out = pool_sum.sum(axis=-1).\
        reshape((batch_size, num_centroids, 2, 2)).dimshuffle(0, 3, 2, 1).\
        reshape((batch_size, 4*num_centroids))          # batch x 4*K

    index = T.iscalar()
    encode = function(inputs=[index], outputs=pool_out,
                      givens={im: X[(index*batch_size):((index+1)*batch_size)]})

    # Main loop
    t0 = time.time()
    features = []
    for k in xrange(n_chunks):
        start = chunk_size*k
        stop = chunk_size*(k+1)
        if k > 0:
            img = np.float32(image[start:stop])
            #pad with zeros
            if img.shape[0]<chunk_size:
                pad = np.zeros((chunk_size-img.shape[0], img.shape[1]), dtype=img.dtype)
                img = np.vstack([img, pad])
            X.set_value(img.reshape(-1, n_channels, imSize, imSize), borrow=True)

        features_chunk = np.vstack([encode(i) for i in xrange(n_batches)])

        if fo is None:
            features.append(features_chunk)
        else:
            # dump to file
            if k == n_chunks-1 and N != n_chunks*chunk_size:
                X_dataset[start:N] = features_chunk[:np.mod(N, chunk_size)]
            else:
                X_dataset[start:stop] = features_chunk

        print 'Encoder: chunk %d/%d' % (k+1, n_chunks)

        
    t1 = time.time()
    print 'Elapsed %d seconds' % (t1 - t0)

    if fo is None:
        return np.vstack(features)[:N]


def encoder_slow(X, centroids, W, M, batch_size=1000):

    '''
    X :         N x D
    centroids : K x d
    W :         d x d
    M :         1 x d
    '''
    n_channels = 3
    num_images = X.shape[0]
    imSize = np.int(np.sqrt(X.shape[1]/n_channels))
    patch_size = np.int(np.sqrt(centroids.shape[1]/n_channels))
    num_centroids = centroids.shape[0]
    epsilon = 10
    XC = np.zeros((num_images, 4*num_centroids))

    half = np.ceil((imSize-patch_size+1)/2.0)

    cc = np.square(centroids).sum(axis=1, keepdims=True).T  # 1 x K
    batch = np.zeros((batch_size, imSize-patch_size+1, imSize-patch_size+1, num_centroids))    
    for b in xrange(0, num_images, batch_size):
        print 'Batch %d/%d' % (b/batch_size + 1 , np.ceil(1.0*num_images/batch_size))
        r = xrange(b, min(b+batch_size, num_images))

        img = X[r].reshape(-1, n_channels, imSize, imSize)
        
        for i in xrange(imSize-patch_size+1):
            for j in xrange(imSize-patch_size+1):
                xx = img[:, :, i:(i+patch_size), j:(j+patch_size)]

                xx = xx.reshape(-1, patch_size*patch_size*n_channels)

                # Normalize the brightness and contrast separately for each patch. 
                mean_ = np.mean(xx, axis=1, dtype='float64', keepdims=True)
                var_ = np.var(xx, axis=1, dtype='float64', ddof=1, keepdims=True)
                xx = (xx-mean_)/np.sqrt(var_+epsilon)

                # Whitening
                xx = np.dot(xx-M, W)                           # N x d

                cx = np.dot(xx, centroids.T)                   # N x K
                xx = np.square(xx).sum(axis=1, keepdims=True)  # N x 1
                distance = np.sqrt(cc - 2*cx + xx)             # N x K
                distance[distance<0] = 0                       # overflow
                batch[:len(r), i, j, :] = distance.mean(axis=1, keepdims=True) - distance

        batch[batch < 0] = 0
        q1 = batch[:len(r), :half, :half, :].sum(axis=(1, 2))
        q2 = batch[:len(r), half:, :half, :].sum(axis=(1, 2))
        q3 = batch[:len(r), :half, half:, :].sum(axis=(1, 2))
        q4 = batch[:len(r), half:, half:, :].sum(axis=(1, 2))
        XC[r] = np.hstack((q1, q2, q3, q4))

    return XC

if __name__ == '__main__':

    mat = scipy.io.loadmat('centroids.mat')
    image = mat['trainX'][:1000]
    centroids = mat['centroids']
    M = mat['M']
    W = mat['PP']

    imSize = np.int(np.sqrt(image.shape[1]/3))
    assert(np.square(imSize)*3 == image.shape[1])

    ################### MATLAB ########################
    image = image.reshape(-1, 3, imSize, imSize).transpose(0, 1, 3, 2).\
        reshape(-1, 3*imSize*imSize)
    psize = np.int(np.sqrt(centroids.shape[1]/3))
    assert(np.square(psize)*3 == centroids.shape[1])
    M = M.reshape(3, psize, psize).transpose(0, 2, 1).reshape(1, 3*psize*psize)
    centroids = centroids.reshape(-1, 3, psize, psize).transpose(0, 1, 3, 2).\
        reshape(-1, 3*psize*psize)
    W = W.reshape(3, psize, psize, 3, psize, psize).\
        transpose(0, 2, 1, 3, 5, 4).reshape(3*psize*psize, 3*psize*psize)
    ##################################################

    features = encoder(image, centroids, W, M, chunk_size=798, batch_size=266)

    features_assert = mat['trainXC'][:1000]
    np.testing.assert_almost_equal(features_assert, features,
                                   decimal=4, err_msg='', verbose=True)

    features_slow = encoder_slow(image, centroids, W, M)
    np.testing.assert_almost_equal(features_assert, features_slow,
                                   decimal=12, err_msg='', verbose=True)






def encoder2(image, centroids, W, M, fo=None,
            chunk_size=999, batch_size=333, Y=None):


    N = image.shape[0]

    # chunk to allocate on GPU
    n_chunks = np.int(np.ceil(1.0*N/chunk_size))
    # batch to process in a single pass
    n_batches = chunk_size/batch_size
    assert(n_batches*batch_size == chunk_size)

    n_channels = 3
    imSize = np.int(np.sqrt(image.shape[1]/n_channels))
    assert(np.square(imSize)*3 == image.shape[1])

    num_centroids = centroids.shape[0]
    psize = np.int(np.sqrt(centroids.shape[1]/n_channels))
    assert(np.square(psize)*n_channels == centroids.shape[1])
    h = imSize - psize + 1
    w = imSize - psize + 1

    if fo is not None:
        X_dataset = fo.create_dataset("X", (N, 4*num_centroids),
                                     dtype=config.floatX)
        if Y is not None:
            fo.create_dataset("y", data=Y)

    img = np.float32(image[:chunk_size])
    #pad with zeros
    if img.shape[0]<chunk_size:
        pad = np.zeros((chunk_size-img.shape[0], img.shape[1]), dtype=img.dtype)
        img = np.vstack([img, pad])    
    X = shared(img.reshape(-1, n_channels, imSize, imSize), borrow=True)
    C = shared(np.float32(centroids), borrow=True)
    W = shared(np.float32(W), borrow=True)
    M = shared(np.float32(M), borrow=True, broadcastable=(True, False))

    cc = T.square(C).sum(axis=1, keepdims=True).T  # 1 x K

    im = T.tensor4(dtype=config.floatX)

    eyef = T.eye(psize * psize, psize * psize, dtype=config.floatX)[::-1]
    filts = T.reshape(eyef, (psize * psize, psize, psize))
    filts = T.shape_padleft(filts).dimshuffle((1, 0, 2, 3))

    res = T.zeros((n_channels, batch_size, psize * psize, h, w), dtype=config.floatX)

    for i in xrange(n_channels):
        cur_slice = T.shape_padleft(im[:, i, :, :]).dimshuffle((1, 0, 2, 3))
        res = T.set_subtensor(res[i], conv.conv2d(cur_slice, filts))

    # res ~ (channel, batch, conv, hi, wi) -> (batch, hi, wi, channel, conv)
    # -> (batch, hi*wi, channel*h*w)
    res = res.dimshuffle((1, 3, 4, 0, 2)).\
        reshape((batch_size*h*w, n_channels*psize*psize))

    # Normalize the brightness and contrast separately for each patch.
    epsilon = 10
    mean_ = T.cast(res.mean(axis=1, keepdims=True), config.floatX)
    dof = n_channels*psize*psize  # adjust DOF
    var_ = T.cast(res.var(axis=1, keepdims=True), config.floatX)*dof/(dof-1)
    res = (res-mean_)/T.sqrt(var_+epsilon)

    # Whitening
    res = T.dot(res-M, W)                          # batch*h*w x n_channels*psize*psize

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # normalise to unit length
    #res /= T.sqrt(T.sqr(res).sum(axis=1, keepdims=True))

    cx = T.dot(res, C.T)                           # batch*h*w x K
    res = T.sqr(res).sum(axis=1, keepdims=True)    # batch*h*w x 1
    distance = cc - 2*cx + res                     # batch*h*w x K
    distance *= (distance > 0)                     # precision issue
    distance = T.sqrt(distance)
    batch = distance.mean(axis=1, keepdims=True) - distance
    batch = batch.reshape((batch_size, h, w, num_centroids)).\
        dimshuffle(0, 3, 1, 2)                     # batch x K x h x w
    batch *= (batch > 0)    # ReLU

    index = T.iscalar()
    encode = function(inputs=[index], outputs=batch,
                      givens={im: X[(index*batch_size):((index+1)*batch_size)]})

    # Main loop
    t0 = time.time()
    features = []
    k = 0
    start = chunk_size*k
    stop = chunk_size*(k+1)
    if k > 0:
        img = np.float32(image[start:stop])
        #pad with zeros
        if img.shape[0]<chunk_size:
            pad = np.zeros((chunk_size-img.shape[0], img.shape[1]), dtype=img.dtype)
            img = np.vstack([img, pad])
        X.set_value(img.reshape(-1, n_channels, imSize, imSize), borrow=True)

    features_chunk = encode(0)

    if fo is None:
        features.append(features_chunk)
    else:
        # dump to file
        if k == n_chunks-1 and N != n_chunks*chunk_size:
            X_dataset[start:N] = features_chunk[:np.mod(N, chunk_size)]
        else:
            X_dataset[start:stop] = features_chunk

    print 'Encoder: chunk %d/%d' % (k+1, n_chunks)

        
    t1 = time.time()
    print 'Elapsed %d seconds' % (t1 - t0)

    if fo is None:
        return np.vstack(features)[:N]