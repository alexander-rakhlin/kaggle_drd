import os


root = 'E:\kaggle\Drd'


# DATA
data_dir = os.path.join(root, 'Data')

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
labels_file = os.path.join(data_dir, 'trainLabels.csv')


# Preprocessed images
train_processed_dir = os.path.join(data_dir, 'train_processed')
test_processed_dir = os.path.join(data_dir, 'test_processed')
if not os.path.exists(train_processed_dir):
    os.makedirs(train_processed_dir)
if not os.path.exists(test_processed_dir):
    os.makedirs(test_processed_dir)
train_packed_file = os.path.join(data_dir, 'train_packed.hd5')
test_packed_file = os.path.join(data_dir, 'test_packed.hd5')

train_packed_list = os.path.join(data_dir, 'train_packed_list.csv')
test_packed_list = os.path.join(data_dir, 'test_packed_list.csv') 


# Features
features_dir = os.path.join(root, 'features')
if not os.path.exists(features_dir):
    os.makedirs(features_dir)
train_features_file = os.path.join(features_dir, 'train_features.7.800.hd5')
train_features_scaled_file = os.path.join(features_dir, 'train_features_scaled.7.800.hd5')
train_features_scaled_strat_file = os.path.join(features_dir, 'train_features_scaled_strat.7.800.hd5')

test_features_file = os.path.join(features_dir, 'test_features.7.800.hd5')
test_features_scaled_file = os.path.join(features_dir, 'test_features_scaled.7.800.hd5')


# NN features
features_NN_dir = os.path.join(root, 'features_NN')
if not os.path.exists(features_NN_dir):
    os.makedirs(features_NN_dir)
    
# Predictions
pred_dir = os.path.join(root, 'prediction')
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
pred_dump = os.path.join(pred_dir, 'pred_dump.7.800.12gen.csv')
submission = os.path.join(pred_dir, 'submission.7.800.12gen.fit_vote.csv')


# Images
nChannels = 3
imSize = 180
outside = 256


# kMeans
nCentroids = 200
nPatches = 60000
patchSize = 15


# Encoder
batch_size = 2
chunk_size = 48