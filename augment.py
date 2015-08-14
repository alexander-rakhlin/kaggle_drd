import constants
from os.path import basename, splitext, join
from os import listdir
import random
from PIL import Image, ImageChops, ImageEnhance
from functools import partial
from multiprocessing import Pool, cpu_count, Value
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit

#random.seed(123)


def get_files_paths(dir, ext='.jpeg'):
    return sorted([join(dir, f) for f in listdir(dir) if splitext(f)[1]==ext])


def class_weights():
    reader = csv.reader(open(constants.labels_file, mode='r'))
    next(reader)  # skip header row
    labels_dict = {row[0]:int(row[1]) for row in reader}
    labels = labels_dict.values()
    label_counts = [sum(1 for p in labels if p==i) for i in range(max(labels)+1)]
    weights = [max(label_counts) / p for p in label_counts]
    return labels_dict, weights


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -10)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def augment_image(im_in, out_dir, N, verbose=False, inside=constants.imSize,
                  outside=constants.outside):

    img = Image.open(im_in)
    
    img = trim(img)
    scale = 1.0 * outside / min(img.size)
    img = img.resize(tuple([int(i*scale) for i in img.size]), Image.BICUBIC)
    
    for n in range(N):
        
        im = img if N==1 else img.copy()
        
        angle = random.uniform(-20, 20)
        mirror = random.randint(0, 1)
        color = random.uniform(0.8, 1.5)
        contrast = random.uniform(0.8, 1.5)
        brightness = random.uniform(0.8, 1.5)
        if verbose:
            print('angle=%d\nmirror=%d\ncolor=%0.1f\ncontrast=%0.1f\nbrightness=%0.1f' %
                 (angle, mirror, color, contrast, brightness))
                 
        if mirror == 1:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = im.rotate(angle)
        z = [(i-inside)/2 for i in im.size]
        im = im.crop((z[0], z[1], z[0]+inside, z[1]+inside))
        
        enhancer = ImageEnhance.Color(im)
        im = enhancer.enhance(color)
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(im)
        im = enhancer.enhance(brightness)
        
        im_out = join(out_dir, splitext(basename(im_in))[0] + '.' +
                      format(n,'02d') + '.png')        
        im.save(im_out, 'PNG')


# auxiliary funciton 
counter = None

def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args
    
    
def augment_image_helper(out_dir, v, total, args):
    global counter
    im_in, N = args
    augment_image(im_in, out_dir, N, verbose=v)
    with counter.get_lock():
        counter.value += 1
        if counter.value % 10 == 0:
            print('%d/%d' % (counter.value, total))


def process_dir(in_dir, out_dir, repeats):
    raw_files = get_files_paths(in_dir)
    if isinstance(repeats, dict):
        job_args = [(f, repeats[splitext(basename(f))[0]]) for f in raw_files]
    else:
        job_args = zip(raw_files, [repeats]*len(raw_files))
    #
    # create the pool of workers, ensuring each one receives the counter 
    # as it starts. 
    #
    nCores = cpu_count()
    counter = Value('i', 0)
    pool = Pool(nCores, initializer = init, initargs = (counter, ))    
    part_f = partial(augment_image_helper, out_dir, False, len(job_args))
    pool.map(part_f, job_args)


def pack_dir(in_dir, out_file, file_list, labels_dict=None, seed=None):
    if seed is not None:
        random.seed(seed)
    files = get_files_paths(in_dir, ext='.png')
    random.shuffle(files)

    with open(file_list, 'w') as fo:
        for f in files:
            fo.write(splitext(basename(f))[0]+'\n')
    
    with h5py.File(out_file, "w") as f5:
        X_dataset = f5.create_dataset("X",
                                     (len(files), constants.nChannels*np.square(constants.imSize)),
                                     dtype=np.uint8)
        if labels_dict is not None:
            y_dataset = f5.create_dataset("y", (len(files),), dtype=int)

        for i,f in enumerate(files):
            im=Image.open(f)
            # N x D (N, n_channels*HSize*WSize)
            d = np.asarray(im.getdata()).T.reshape(1,-1)
            X_dataset[i] = np.uint8(d)
            if labels_dict is not None:
                y_dataset[i] = labels_dict[splitext(splitext(basename(f))[0])[0]]
#                print y_dataset[i]
                
            if (i+1) % 50 == 0:
                print('%d/%d' % (i+1, len(files)))

#            print f
#            plt.imshow(X_dataset[i].reshape(3,180,180)[1,:,:])
#            break


def stratify_packed_file(in_dir, fin, fout, labels_dict, test_size=0.15, seed=None):
    if seed is not None:
        random.seed(seed)
    files = get_files_paths(in_dir, ext='.png')
    random.shuffle(files)
    

    names_labels = labels_dict.items()
    names, labels = zip(*names_labels)

    sss = StratifiedShuffleSplit(labels, n_iter=1,
                                 test_size=test_size, random_state=0)
    
    for train_index, test_index in sss:
        for l in range(5):
            print sum([1 for i in train_index if labels[i] == l]), sum([1 for i in test_index if labels[i] == l])
        print len(train_index), len(test_index)
        
    train_index, test_index = list(sss)[0]
    
    with h5py.File(fin, "r") as fi:
        X = fi.get("X")
        y = fi.get("y")
        
        nTest = nTrain = 0
        train_names = []
        test_names = []
        for f in files:
            fname = splitext(splitext(basename(f))[0])[0]
            index = names.index(fname)
            if index in train_index:
                nTrain += 1
                train_names.append(fname)
            elif index in test_index and not(fname in test_names):
                nTest += 1
                test_names.append(fname)
                
        with h5py.File(fout, "w") as fo:

            X_train = fo.create_dataset("X_train", (nTrain, X.shape[1]), dtype=X.dtype)
            y_train = fo.create_dataset("y_train", (nTrain,), dtype=y.dtype)
            X_test = fo.create_dataset("X_test", (nTest, X.shape[1]), dtype=X.dtype)
            y_test = fo.create_dataset("y_test", (nTest,), dtype=y.dtype)
            
            trn = tst = 0
            skipped = 0
            train_names = []
            test_names = []
            for i,f in enumerate(files):
                fname = splitext(splitext(basename(f))[0])[0]
                index = names.index(fname)
                if index in train_index:
                    # append to X_train
                    X_train[trn] = X[i]
                    y_train[trn] = y[i]
                    trn += 1
                    train_names.append(fname)
                elif index in test_index and not(fname in test_names):
                    # append to X_test and remove from names
                    X_test[tst] = X[i]
                    y_test[tst] = y[i]
                    tst += 1
                    test_names.append(fname)
                else:
                    skipped += 1

                if i % 500 == 0:
                    print('%d/%d' % (i, len(files)))
                    
            print 'X_train shape:', X_train.shape
            print 'X_test shape:', X_test.shape
            print 'X shape:', X.shape
            print 'Skipped:', skipped
            
            assert(len(np.unique(test_names)) == len(test_names))
            assert(set.intersection(set(train_names), set(test_names)) == set([]))
            assert(len(train_names) + len(test_names) + skipped == X.shape[0])
            
            assert(all([names.index(n) in train_index for n in train_names]))
            assert(all([names.index(n) in test_index for n in test_names]))
            
            assert(len(np.unique(train_names)) == len(train_index))
            assert(len(test_names) == len(test_index))

        
if __name__ == '__main__':
    
    labels_dict, weights = class_weights()
    print 'weights = ', weights
    
    repeats = {k:weights[v] for k,v in labels_dict.items()}
    process_dir(constants.train_dir, constants.train_processed_dir, repeats)

    pack_dir(constants.train_processed_dir, constants.train_packed_file,
             labels_dict=labels_dict, seed=123)

    repeats = 3
    process_dir(constants.test_dir, constants.test_processed_dir, repeats)

    pack_dir(constants.test_processed_dir, constants.test_packed_file, 
             constants.test_packed_list, seed=123)

    stratify_packed_file(constants.train_processed_dir,
                         constants.train_features_scaled_file,
                         constants.train_features_scaled_strat_file,
                         labels_dict, test_size=0.15, seed=123)