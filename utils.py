import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt
plt.ion()
from PIL import ImageEnhance, Image
import csv
import constants
from scaler import fit2distribution
from pandas import read_csv
from os.path import splitext


def draw_patches(images, H=28, W=28, br = 1, cont = 1, figure=1):
    
   channels=images.shape[1]/(H*W)
   mn=images.min()
   mx=images.max()
   
   K=images.shape[0]
   COLS=np.round(np.sqrt(K))
   ROWS=np.ceil(K / COLS)
   
   image=np.ones((ROWS*(H+1), COLS*(W+1), channels))*mx
   for i in xrange(images.shape[0]):
       r = np.floor(i / COLS)
       c = np.mod(i, COLS)
       tmp = images[i,:].reshape(channels,H,W).transpose(1,2,0)
       image[(r*(H+1)):(r*(H+1)+H),(c*(W+1)):(c*(W+1)+W),:] = tmp
           
   image = np.uint8((image - mn) / (mx - mn) * 255)

   im = Image.fromarray(image, mode=None) 
   if br != 1:
       Brightness = ImageEnhance.Brightness(im)
       im=Brightness.enhance(br)
   if cont != 1:       
       Contrast = ImageEnhance.Contrast(im)
       im=Contrast.enhance(cont)
   
   plt.figure(figure)
   plt.imshow(im)
   plt.show(block=True)


def dump_prediction(level, idfile, fout):
    level = level.flatten()
    ids = open(idfile).read().splitlines()
    with open(fout, 'wb') as fo:
        writer = csv.writer(fo)
        writer.writerow(('image', 'level'))
        zz = zip(ids, map(str, level))
        writer.writerows(zz)
    print("Dumped prediction to file {}".format(fout))
    
    
def make_submission(fin=constants.pred_dump, fout=constants.submission,
                    mode='pullup_fit'):
    '''
    mode = fit_vote | pullup_fit | classify_vote
    '''
    df = read_csv(fin)
    df['image'] = df['image'].map(lambda x: splitext(x)[0])

    def f(x):
        xs = sorted(x, reverse=True)
        return xs[np.argmax([xs.count(l) for l in xs])]

    if mode == 'fit_vote':
        df_labels = read_csv(constants.labels_file)
        df['level'] = fit2distribution(df_labels['level'].values, df['level'].values)
        r = df.groupby('image').agg(lambda x: f(x.tolist()))
        
    elif  mode == 'pullup_fit':
        df_labels = read_csv(constants.labels_file)
        r = df.groupby('image').max()
        r['level'] = fit2distribution(df_labels['level'].values, r['level'].values)

    elif  mode == 'classify_vote':
        y = np.int_((df['level'].values-0.5)*2*5.0)
        y[y<0] = 0
        y[y>4] = 4
        df['level'] = y
        r = df.groupby('image').agg(lambda x: f(x.tolist()))
    
    r.to_csv(fout)
            
    with zipfile.ZipFile(fout+'.zip', 'w', zipfile.ZIP_DEFLATED) as fo:
        fo.write(fout, arcname=os.path.basename(fout))
    os.remove(fout)
    print("Wrote submission to file {}".format(fout+'.zip'))
    
    
if __name__ == '__main__':
    make_submission(mode='fit_vote')



