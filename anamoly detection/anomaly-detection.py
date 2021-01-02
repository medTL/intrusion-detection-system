dataset_base_path = r'../../cic-ids-2018'
import os
import sys
sys.path.append('/media/talel/TOSHIBA EXT/ids/')
os.getcwd()
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from ml_ids.data.dataset import load_dataset
from ml_ids.transform.preprocessing import create_pipeline
from ml_ids.model_selection import split_x_y
from ml_ids.visualization import plot_confusion_matrix, plot_hist, plot_threshold
from ml_ids.keras.metrics import AveragePrecisionScoreMetric
from ml_ids.keras.callbacks import OneCycleScheduler
from ml_ids.libs.dfencoder.dataframe import EncoderDataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.utils.multiclass import unique_labels
from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers, constraints
from scipy import stats
from notebook_utils import predict, evaluate_pr_roc, plot_evaluation_curves, plot_pr_threshold_curves, best_precision_for_target_recall, print_performance, filter_benign

K = keras.backend

rand_state = 42
tf.random.set_seed(rand_state)
np.random.seed(rand_state)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

cols_to_impute = [
    'flow_duration',
    'flow_byts_s',
    'flow_pkts_s',
    'flow_iat_mean',
    'flow_iat_max',
    'flow_iat_min',
    'fwd_iat_tot',
    'fwd_iat_mean',
    'fwd_iat_max',
    'fwd_iat_min'
]

features_no_variance = [
    "bwd_blk_rate_avg",
    "bwd_byts_b_avg",
    "bwd_pkts_b_avg",
    "bwd_psh_flags",
    "bwd_urg_flags",
    "fwd_blk_rate_avg",
    "fwd_byts_b_avg",
    "fwd_pkts_b_avg"
]

features_same_distribution = [
    'fwd_urg_flags',
    'pkt_len_var',
    'fin_flag_cnt',
    'urg_flag_cnt',
    'cwe_flag_count',
    'down_up_ratio',
    'fwd_act_data_pkts',
    'active_max'
]

omit_cols = features_no_variance + features_same_distribution + ['timestamp', 'dst_port', 'protocol']

ids_data = load_dataset(dataset_base_path,
                        omit_cols=omit_cols,
                        preserve_neg_value_cols=['init_fwd_win_byts', 'init_bwd_win_byts'])

benign_mask = ids_data.label_is_attack == 0
attack_mask = ids_data.label_is_attack == 1

train_data, hold_data = train_test_split(ids_data[benign_mask], test_size=0.2, random_state=rand_state)

val_data_benign, test_data_benign = train_test_split(hold_data, test_size=0.5, random_state=rand_state)

val_data_attack, test_data_attack = (train_test_split(ids_data[attack_mask],
                                                      test_size=0.5,
                                                      stratify=ids_data[attack_mask].label_cat,
                                                      random_state=rand_state))


X_train_raw, y_train = split_x_y(train_data)
X_val_raw, y_val = split_x_y(val_data_benign.append(val_data_attack))
X_test_raw, y_test = split_x_y(test_data_benign.append(test_data_attack))

print('Samples:')
print('========')
print('Training: {}'.format(X_train_raw.shape))
print('Val:      {}'.format(X_val_raw.shape))
print('Test:     {}'.format(X_test_raw.shape))

print('\nTraining labels:')
print('================')
print(y_train.label.value_counts())
print('\nValidation labels:')
print('==================')
print(y_val.label.value_counts())
print('\nTest labels:')
print('============')
print(y_test.label.value_counts())

del ids_data, train_data, hold_data, val_data_benign, val_data_attack, test_data_benign, test_data_attack
gc.collect()

pipeline, get_col_names = create_pipeline(X_train_raw, 
                                          imputer_strategy='median',
                                          imputer_cols=cols_to_impute,
                                          scaler=MinMaxScaler)

X_train = pipeline.fit_transform(X_train_raw)
X_val = pipeline.transform(X_val_raw)
X_test = pipeline.transform(X_test_raw)

X_val_benign = filter_benign(X_val, y_val)

column_names = get_col_names()

print('Samples:')
print('========')
print('Training: {}'.format(X_train.shape))
print('Val:      {}'.format(X_val.shape))
print('Test:     {}'.format(X_test.shape))

print('\nMissing values:')
print('===============')
print('Training: {}'.format(np.count_nonzero(np.isnan(X_train))))
print('Val:      {}'.format(np.count_nonzero(np.isnan(X_val))))
print('Test:     {}'.format(np.count_nonzero(np.isnan(X_test))))

print('\nScaling:')
print('========')
print('Training: min={}, max={}'.format(np.min(X_train), np.max(X_train)))
print('Val:      min={}, max={}'.format(np.min(X_val), np.max(X_val)))
print('Test:     min={}, max={}'.format(np.min(X_test), np.max(X_test)))

input_dims = X_train.shape[1]
epochs = 50
batch_size = 16

K.clear_session()
gc.collect()

simple_ae = models.Sequential([
    layers.Dense(30, activation='elu', input_shape=[input_dims]),
    layers.Dense(input_dims, activation='sigmoid')
])

simple_ae.compile(optimizer='adam', loss='binary_crossentropy')
simple_ae.summary()

early_stopping = callbacks.EarlyStopping(monitor='val_auprc', 
                                         mode='max',
                                         patience=15,                             
                                         restore_best_weights=True)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_auprc', 
                                        mode='max', 
                                        factor=0.2, 
                                        patience=3, 
                                        min_lr=0.0001)

mc = callbacks.ModelCheckpoint(filepath='models/simple_autoencoder_model.h5',
                               monitor='val_auprc', 
                               mode='max',
                               save_best_only=True, 
                               verbose=0)

hist = simple_ae.fit(x=X_train, 
                     y=X_train, 
                     validation_data=(X_val_benign, X_val_benign),
                     batch_size=batch_size,
                     epochs=epochs,
                     callbacks=[
                         AveragePrecisionScoreMetric(X_val=X_val, y_val=y_val.label_is_attack, batch_size=16384),
                         early_stopping,
                         reduce_lr,
                         mc
                     ])