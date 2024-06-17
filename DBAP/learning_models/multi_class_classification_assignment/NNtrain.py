#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 21:53:45 2022
@author: korekane
train NN
"""


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.metrics import CategoricalAccuracy
# from tensorflow_addons.metrics import F1Score
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_data(path_r, berth_num):
    x = []
    y = []
    path = path_r+".csv"
    data_num = sum([1 for _ in open(path)])
    with open(path) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        for i in range(len(l)):
            if "" in l[i]:
                l[i].remove("")
        text = [[int(v) for v in row] for row in l]
    for index in range(data_num):
        x.append(text[index][:-1])
    for index in range(data_num):
        y.append(np.eye(berth_num)[text[index][-1]])
    return x, y

def build_model(berth_num, alpha):
    layyer_1 = 3
    input_data = []
    x1 = []
    dense = []
    for i in range(layyer_1):
        dense.append(Dense(32, activation='relu'))
    for index_1 in range(berth_num):
        # input_data.append(Input(shape = (14,)))
        input_data.append(Input(shape = (alpha,)))
        x1.append(dense[0](input_data[index_1]))
        for index_2 in range(layyer_1 - 1):
            x1[index_1] = dense[index_2 + 1](x1[index_1])

    z = concatenate(x1, axis=-1)

    layyer_2 = 3
    z = Dense(64, activation = "relu")(z)
    for index in range(layyer_2-1):
        z = Dense(32, activation = "relu")(z)
    z = Dense(berth_num, activation = "softmax")(z)
    model = Model(inputs = input_data, outputs = z)
    learning_rate = 0.001
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer = adam,
                  # metrics=['accuracy', F1Score(num_classes = berth_num, average = 'macro', name = 'f1_score')])
                   metrics=[CategoricalAccuracy(name = 'accuracy')])
    return model

def train():
    # ship_num = 60
    berth_num = 13
    alpha = 11
    
    path_r="train(3000)"
    print("============load data============")
    x_train , y_train = load_data(path_r, berth_num)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    input_train = []
    for index in range(berth_num):
        x, x_train = np.split(x_train,[14,],1)
        z = np.delete(x, list(range(alpha, 14)), axis = 1)
        input_train.append(z)
    print("=============complete==============")
    
    print("============build model============")
    model = build_model(berth_num, alpha)
    print("=============complete==============")
    nb_epoch = 100
    batch_size = 128
    
    print("=====learning=====")
    result = model.fit(input_train,
                       y_train,
                       batch_size = batch_size,
                       epochs = nb_epoch,
                       verbose = 1,
                       validation_split = 0.2
                       )
    print("=====complete=====")
    
    path_r = "train(3000)"
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in'  
    plt.rcParams["font.size"] = 10
    result.history.keys()
    
    fig = plt.figure()
    fig_1 = fig.add_subplot(111)
    newlist = [n*100 for n in result.history['accuracy']]
    fig_1.plot(range(1, nb_epoch+1), newlist, label="training")
    newlist = [n*100 for n in result.history['val_accuracy']]
    fig_1.plot(range(1, nb_epoch+1), newlist, label="validation")
    fig_1.set_xlabel('Epochs[times]',fontsize=14)
    fig_1.set_ylabel('Accuracy[%]',fontsize=14)
    fig_1.legend(loc=0)
    fig.savefig('fig2/'+path_r+'_acc.eps', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    fig.savefig('fig2/'+path_r+'_acc.svg', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    fig.savefig('fig2/'+path_r+'_acc.png', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    fig.savefig('fig2/'+path_r+'_acc.jpg', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    
    # fig = plt.figure()
    # fig_1 = fig.add_subplot(111)
    # newlist = [n*100 for n in result.history['f1_score']]
    # fig_1.plot(range(1, nb_epoch+1), newlist, label="training")
    # newlist = [n*100 for n in result.history['f1_score']]
    # fig_1.plot(range(1, nb_epoch+1), newlist, label="validation")
    # fig_1.set_xlabel('Epochs[times]',fontsize=14)
    # fig_1.set_ylabel('F1 score[%]',fontsize=14)
    # fig_1.legend(loc='upper left')
    # fig.savefig('fig2/'+path_r+'_f1_score.eps', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    # fig.savefig('fig2/'+path_r+'_f1_score.svg', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    # fig.savefig('fig2/'+path_r+'_f1_score.png', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    # fig.savefig('fig2/'+path_r+'_f1_score.jpg', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    
    fig = plt.figure()
    fig_2 = fig.add_subplot(111)
    fig_2.plot(range(1, nb_epoch+1),result.history['loss'], label="training")
    fig_2.plot(result.history['val_loss'], label="validation")
    f = open('myfile.txt', 'w')
    fig_2.set_xlabel('Epochs[times]',fontsize=14)
    fig_2.set_ylabel('Loss[-]',fontsize=14)
    fig_2.legend(loc=0)
    fig.savefig('fig2/'+path_r+'_loss.eps', bbox_inches = "tight", pad_inches = 0.05, dpi = 300)
    fig.savefig('fig2/'+path_r+'_loss.svg', bbox_inches = "tight", pad_inches = 0.05, dpi = 300)
    fig.savefig('fig2/'+path_r+'_loss.png', bbox_inches = "tight", pad_inches = 0.05, dpi = 300)
    fig.savefig('fig2/'+path_r+'_loss.jpg', bbox_inches = "tight", pad_inches = 0.05, dpi = 300)
    
    with open('fig_value2/'+path_r+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(result.history['accuracy'])
        writer.writerow(result.history['val_accuracy'])
        # writer.writerow(result.history['f1_score'])
        # writer.writerow(result.history['val_f1_score'])
        writer.writerow(result.history['loss'])
        writer.writerow(result.history['val_loss'])
    model.save('model2/'+path_r+'.h5')


def main():
    train()

if __name__ == "__main__":
    main()