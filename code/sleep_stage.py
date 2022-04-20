# -*- coding: utf-8 -*-
""" 
@Time    : 2021/11/27 22:45
@Author  : Yihong Lu, Ziye Zheng, Lizhu Wu, Jiahui Pan
@FileName: sleep_stage.py
@SoftWare: PyCharm
"""

import urllib
from scipy.signal import spectrogram
import pickle
import random
from librosa.feature import melspectrogram
from librosa import power_to_db
from librosa.display import specshow
import urllib.request
from pyedflib import highlevel
import glob
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import datetime
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.regularizers import l2
from keras.layers import Reshape, Flatten, TimeDistributed, Bidirectional, BatchNormalization, Dropout, Input, Add, Masking, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Activation
from keras import Model
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.models import load_model

# sys.executable


# all EEG data is 100Hz sampling rate
FS = 100
fs=100
# length of spectrogram in seconds
SPEC_LEN = 30

# download Sleep-EDF URL: 'https://archive.physionet.org/physiobank/database/sleep-edfx/'

#local path
data_dir = 'data/'

#load data
# list of all files
hypnogram_files = glob.glob('%s/*-Hypnogram.edf' % data_dir)
psg_files = glob.glob('%s/*-PSG.edf' % data_dir)

# hypnogram_files: data/SCxxxxxx-Hypnogram.edf
for f in hypnogram_files:
    file = f.split('\\')[1]
    file_name = file.split('.')[0]
    hypnogram_filepath = f

    # get the corresponding PSG filepath
    file_id = hypnogram_filepath.split('\\')[1][0:7]
    file_index = [file_id in psg_file for psg_file in psg_files]
    file_index = np.where(np.array(file_index) == True)[0][0]
    psg_filepath = psg_files[file_index]  # find response PSG file

# # an example
# psg_filepath='data/SC4002E0-PSG.edf'
# hypnogram_filepath='data/SC4002EC-Hypnogram.edf'
try:
        psg_signals, psg_signal_headers, psg_header = highlevel.read_edf(psg_filepath)
        hypnogram_signals, hypnogram_signal_headers, hypnogram_header = highlevel.read_edf(hypnogram_filepath)
except Exception as e:
        print(str(e))
        #continue

    # put into a dataframe
psg_columns = [psg_signal_headers[i]['label'] for i in range(2)]  # two EEG channels: ‘EEG Fpz-Cz’ and ‘EEG Pz-Oz’
psg_df = pd.DataFrame({psg_columns[0]: psg_signals[0], psg_columns[1]: psg_signals[1]})
    # print(psg_df)
    # generate a sample resolution sleep stage label
stages = []
for annotation in hypnogram_header['annotations']:
        dur = int(str(annotation[1])[2:-1])
        sleep_stage = annotation[2][-1]

        if sleep_stage == 'W':
            sleep_stage = 0
        elif sleep_stage == '1':
            sleep_stage = 1
        elif sleep_stage == '2':
            sleep_stage = 2
        elif sleep_stage == '3':
            sleep_stage = 3
        elif sleep_stage == '4':
            sleep_stage = 4
        elif sleep_stage == 'R':
            sleep_stage = 5
        else:
            sleep_stage = -1
        stages.extend([sleep_stage for i in range(dur * fs)])

    # cut off the psg data at the length of the sleep stage labels
if len(psg_df) > len(stages):
    psg_df = psg_df[0:len(stages)]
else:
    stages = stages[0:len(psg_df)]

print("sleep_stages")
print(len(stages))
psg_df = psg_df.assign(label=stages)  # 增加一列
psg_df.to_pickle(hypnogram_filepath.split('-')[0] + '.pkl')


# spectrograms

# define Mel Spectrogram parameters
n_fft = 256
hop_length = 64
n_mels = 64

def get_most_frequent_label(labels):
    unique_labels = np.unique(labels)
    label_counts = {}
    for unique_label in unique_labels:
        label_counts[unique_label] = sum(labels==unique_label)

    most_frequent_label = unique_labels[0]
    for key, value in label_counts.items():
        if value > label_counts[most_frequent_label]:
            most_frequent_label = key

    return most_frequent_label

def calculate_spectrograms(pkl_files, data_group):

    spectrogram_list = []
    labels_list = []

    for pkl_file in pkl_files:
        df = pd.read_pickle(pkl_file)  # ['EEG Fpz-Cz', 'EEG Pz-Oz', 'label']
        spectrogram_list_tmp = []
        labels_list_tmp = []

        ind = 0
        while(ind < len(df)):
            df_tmp = df.iloc[ind:ind + FS * SPEC_LEN]
            ch1_tmp = df_tmp['EEG Fpz-Cz'].values
            ch2_tmp = df_tmp['EEG Pz-Oz'].values
            label_tmp = get_most_frequent_label(df_tmp['label'].values)
            try:
                # (3000,)
                ch1_tmp = (ch1_tmp - np.mean(ch1_tmp)) / np.std(ch1_tmp)
                # (3000,)
                ch2_tmp = (ch2_tmp - np.mean(ch2_tmp)) / np.std(ch2_tmp)
            except Exception as e:
                spectrogram_list_tmp = []
                labels_list_tmp = []
                ind = ind + FS * SPEC_LEN
                continue

            # calculate spectrograms

            f, t, Sxx1 = spectrogram(ch1_tmp, fs=1.0, window=('tukey', 0.25))
            Sxx1 = power_to_db(Sxx1, ref=np.max)
            # (129, 13)
            f, t, Sxx2 = spectrogram(ch2_tmp, fs=1.0, window=('tukey', 0.25))
            # (129, 13)
            Sxx2 = power_to_db(Sxx2, ref=np.max)

            # (array([[...], [...], ...]), array([[...], [...], ...]))
            spectrogram_list_tmp.append((Sxx1, Sxx2))
            labels_list_tmp.append(label_tmp)
            ind = ind + FS * SPEC_LEN

            if len(spectrogram_list_tmp) == 5:
                # package this result
                if -1 not in labels_list_tmp:
                    spectrogram_list.append(spectrogram_list_tmp)
                    labels_list.append(labels_list_tmp)

                spectrogram_list_tmp = []
                labels_list_tmp = []

    with open('X_%s_spec.pkl' % data_group, 'wb') as f:
        pickle.dump(spectrogram_list, f)
    with open('y_%s.pkl' % data_group, 'wb') as f:
        pickle.dump(labels_list, f)

# pick a random 80% of the files for training
train = []
# pick a random 10% of the files for validation
val = []
# pick a random 10% of the files for testing
test = []
calculate_spectrograms(train, 'train')
calculate_spectrograms(val, 'val')
calculate_spectrograms(test, 'test')

# train model

BATCH_SIZE = 32
SPECTROGRAM_LEN = 13
SPECTROGRAM_FREQS = 129
NUM_OUTPUT_CLASSES = 4
CONV_OUTPUT_LEN = 30
shared_conv1 = Conv2D(filters=8, kernel_size=3, padding='same')
shared_bn1 = BatchNormalization()
shared_relu1 = Activation('relu')
shared_mp1 = MaxPooling2D(pool_size=4, strides=4)
shared_do1 = Dropout(0.2)
shared_conv2 = Conv2D(filters=16, kernel_size=3, padding='same')
shared_bn2 = BatchNormalization()
shared_relu2 = Activation('relu')
shared_mp2 = MaxPooling2D(pool_size=2, strides=2)
shared_do2 = Dropout(0.2)
shared_flatten = Flatten()
shared_dense1 = Dense(units=CONV_OUTPUT_LEN)
shared_bn4 = BatchNormalization()
shared_relu4 = Activation('relu')

def load_data(data_group):
    data = pickle.load(open('X_%s_spec.pkl' % data_group, 'rb'))
    labels = pickle.load(open('y_%s.pkl' % data_group, 'rb'))

    # combine NREM1, NREM2 into LS class, combine NREM3, NREM4 into SWS class
    # new class definitions:
    # Wake = 0 originally, this remains the same
    # N1 = 1, N2 = 2 originally, now they are all assigned to the label of 1
    # N3 = 3, N4 = 4 originally, now they are all assigned to the label of 2
    # REM = 5 originally, now REM = 3
    # Artifact/Unlabeled: N/A as any such labels were already removed in spectrogram generation

    four_class_labels = []
    for label in labels:
        label = np.array(label)
        label_2_inds = np.where(label == 2)[0]
        label_3_inds = np.where(label == 3)[0]
        label_4_inds = np.where(label == 4)[0]
        label_5_inds = np.where(label == 5)[0]

        label[label_2_inds] = 1
        label[label_3_inds] = 2
        label[label_4_inds] = 2
        label[label_5_inds] = 3

        four_class_labels.append(label)

    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []
    y = []
    # y =
    for i in range(len(data)):
        X1_tmp = data[i][0]
        X2_tmp = data[i][1]
        X3_tmp = data[i][2]
        X4_tmp = data[i][3]
        X5_tmp = data[i][4]


        if (np.shape(X1_tmp)[2] != SPECTROGRAM_LEN) or (np.shape(X2_tmp)[2] != SPECTROGRAM_LEN) or (np.shape(X3_tmp)[2] != SPECTROGRAM_LEN) or (np.shape(X4_tmp)[2] != SPECTROGRAM_LEN) or(np.shape(X5_tmp)[2] != SPECTROGRAM_LEN):
           continue

        X1.append(X1_tmp)
        X2.append(X2_tmp)
        X3.append(X3_tmp)
        X4.append(X4_tmp)
        X5.append(X5_tmp)
        y.append(four_class_labels[i])

    X1 = np.array(X1).reshape(len(X1), SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2)
    X2 = np.array(X2).reshape(len(X2), SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2)
    X3 = np.array(X3).reshape(len(X3), SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2)
    X4 = np.array(X4).reshape(len(X4), SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2)
    X5 = np.array(X5).reshape(len(X5), SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2)


    y = to_categorical(y)

    return X1, X2, X3, X4, X5, y

def performance_metrics1(predictions):
    predictions_class = []
    for prediction in predictions:
        for row in prediction:
            predictions_class.append(np.argmax(row))

    print("\n")
    print(predictions_class)

def performance_metrics(true, predictions, labels):

    # loop over each row, and then get the index corresponding to the max probability
    predictions_class = []
    for prediction in predictions:
        for row in prediction:
            predictions_class.append(np.argmax(row))

    # repeat for the true labels
    true_class = []
    for true_row in true:
        for row in true_row:
            true_class.append(np.argmax(row))

    # calculate confusion matrix
    conf_mat = confusion_matrix(true_class, predictions_class)
    conf_mat = pd.DataFrame(conf_mat, columns=labels, index=labels)

    # calculate per class f1
    f1 = f1_score(true_class, predictions_class, average=None)
    f1 = pd.Series(f1, index=labels)

    return conf_mat, f1, predictions_class, true_class


def build_cnn(input):

    cnn_output0 = build_cnn_per_channel(input, 0)
    cnn_output1 = build_cnn_per_channel(input, 1)

    # concatenate each channel's output
    cnn_output = Concatenate(axis=1)([cnn_output0, cnn_output1])
    print(cnn_output.shape)
    cnn_output = Reshape((1, CONV_OUTPUT_LEN * 2))(cnn_output)
    print(cnn_output.shape)
    return cnn_output


def build_cnn_per_channel(input, ch):
    cnn_output = shared_conv1(input[:, :, :, ch][..., None])
    cnn_output = shared_bn1(cnn_output)
    cnn_output = shared_relu1(cnn_output)
    cnn_output = shared_mp1(cnn_output)
    cnn_output = shared_do1(cnn_output)
    cnn_output = shared_conv2(cnn_output)
    cnn_output = shared_bn2(cnn_output)
    cnn_output = shared_relu2(cnn_output)
    cnn_output = shared_mp2(cnn_output)
    cnn_output = shared_do2(cnn_output)
    cnn_output = shared_flatten(cnn_output)
    cnn_output = shared_dense1(cnn_output)
    cnn_output = shared_bn4(cnn_output)
    cnn_output = shared_relu4(cnn_output)
    cnn_output = Reshape((1, CONV_OUTPUT_LEN))(cnn_output)

    return cnn_output

def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)

    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed


X1_train, X2_train, X3_train, X4_train, X5_train, y_train = load_data('train')
X1_val, X2_val, X3_val, X4_val, X5_val, y_val = load_data('value')

# calculate the weights of each class
from sklearn.utils import class_weight
yyt=[]
for yy in y_train:
    for row in yy:
       yyt.append(np.argmax(row))

weight = class_weight.compute_class_weight('balanced', np.unique(yyt), yyt)
weight = {i : weight[i] for i in range(4)}
print(weight)
weight=np.array(weight)

input1 = Input(shape=(SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2))
input2 = Input(shape=(SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2))
input3 = Input(shape=(SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2))
input4 = Input(shape=(SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2))
input5 = Input(shape=(SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2))

cnn_output1 = build_cnn(input1)
cnn_output2 = build_cnn(input2)
cnn_output3 = build_cnn(input3)
cnn_output4 = build_cnn(input4)
cnn_output5 = build_cnn(input5)

# combine all five outputs into a single input for the LSTM
cnn_concat = Concatenate(axis=1)([cnn_output1, cnn_output2, cnn_output3, cnn_output4, cnn_output5])

lstm_output = Bidirectional(LSTM(15, return_sequences=True, kernel_regularizer=l2(0.01)))(cnn_concat)
lstm_output = Dense(NUM_OUTPUT_CLASSES, activation='softmax')(lstm_output)

model = Model(inputs=[input1, input2, input3, input4, input5], outputs=lstm_output)
model.compile(loss=[multi_category_focal_loss1(alpha=[[0.36],[1.26],[5.73],[3.68]], gamma=2)], metrics=['categorical_accuracy'], optimizer='adam')

print(model.summary())

##test model
model = load_model('model.h5',compile = False)

X1_test, X2_test, X3_test, X4_test, X5_test, y_test = load_data('mydata')
predictions = model.predict([X1_test, X2_test, X3_test, X4_test, X5_test])
print("------------------------------")
print(predictions)
conf_mat, f1, _, _ = performance_metrics(y_test, predictions, ['Wake', 'LS', 'SWS', 'REM'])
print('Confusion Matrix:')
print(conf_mat)

print('\nf1 score, per class:')
print(f1)
conf=np.array(conf_mat)
# 精度
tol=np.sum(conf)
pe_rows = np.sum(conf_mat, axis=0)
pe_cols = np.sum(conf_mat, axis=1)
accurracy = (conf[0, 0] + conf[1, 1]+conf[2, 2]+conf[3, 3]) /tol
print("accurracy:")
print(accurracy)
print("W_acc:")
a1=conf[0,0]/pe_cols[0]
print(a1)
print("LS_acc:")
a2=conf[1,1]/pe_cols[1]
print(a2)
print("SWS_acc:")
a3=conf[2,2]/pe_cols[2]
print(a3)
print("REM_acc:")
a4=conf[3,3]/pe_cols[3]
print(a4)
pe_rows = np.sum(conf_mat, axis=0)
pe_cols = np.sum(conf_mat, axis=1)
sum_total = sum(pe_cols)
pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
po = np.trace(conf_mat) / float(sum_total)
print("kappa:")
print((po - pe) / (1 - pe))