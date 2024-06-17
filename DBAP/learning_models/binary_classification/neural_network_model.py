from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Recall, Precision
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import pandas as pd
import time


def load_data(path_r):
    #入力データのファイルの読み込み
    x_train = []
    y_train = []
    path = "learn/"+path_r+".csv"
    with open(path) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        text = [[int(v) for v in row] for row in l]
        data_num = len(text)
        # シャッフル
        # random.shuffle(text)
    for index in range(data_num):#データ数を代入
        x_train.append(text[index][:-1])
    #出力データのファイル読み込み
    for index in range(data_num):#データ数を代入
        y_train.append(text[index][-1])
    return x_train, y_train

def build_model(ship_num, berth_num, max_time_frame):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(126,)))
    # model.add(Dense(64, activation="relu", input_shape=(128,)))
    # model.add(Dense(64, activation="relu", input_shape=(690,)))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy', 
                           Recall(class_id = None, name="recall"), 
                           Precision(class_id = None, name="precision")])
    return model

def deep_learn():
    
    ship_num = 10
    berth_num = 4
    max_time_frame = 14
    
    #path_r="sample"
    #path_r='ship'+str(ship_num)+'_berth'+str(berth_num)
    #path_r='ship10_berth4(100)'
    
    path_r='s10b4x100'
    # path_r='s10b4x3000'
    """
    ship_num = 15
    berth_num = 3
    max_time_frame = 200
    path_r='learn_data(f15x3)-100'
    """
    
    """
    #データセットの数
    x_train , y_train = load_data(path_r)
    #numpyに変換
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x1_train, x2_train, x3_train, x4_train = np.split(x_train,
                                                                [
                                                                 berth_num * max_time_frame,
                                                                 berth_num * max_time_frame + ship_num*2,
                                                                 berth_num * max_time_frame + ship_num*3,
                                                                 ],
                                                                1)

    #入力の正規化
    x2_train = x2_train / max_time_frame
    x3_train = x3_train / ship_num
    x4_train = x4_train / max_time_frame
    train = np.concatenate([x1_train, x2_train, x3_train, x4_train], 1)
    """
    
    # learn_data = np.loadtxt("learn/"+path_r+".csv")
    df = pd.read_csv("learn/"+path_r+".csv", header=None)
    df = df.values
    
    train = df[:, :-1]
    print(train.shape)
    
    y_train = df[:, -1]
    print(y_train.shape)
    
    
    #引数にデータ数を打ち込む
    model = build_model(ship_num, berth_num, max_time_frame)
    
    nb_epoch = 100
    batch_size = 256
    class_weight = {0: 1.,
                    1: 1.}
    
    t = time.time()
    result = model.fit([train],
                       y_train,
                       batch_size = batch_size,
                       epochs = nb_epoch,
                       class_weight=class_weight,
                       validation_split = 0.2)
    print(time.time()-t)
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams["font.size"] = 10

    path_r=str(3)+path_r

    result.history.keys()# ヒストリデータのラベルを見てみる
    fig = plt.figure()
    fig_1 = fig.add_subplot(111)
    newlist = [n*100 for n in result.history['accuracy']]
    fig_1.plot(range(1, nb_epoch+1), newlist, label="training")
    newlist = [n*100 for n in result.history['val_accuracy']]
    fig_1.plot(range(1, nb_epoch+1), newlist, label="validation")
    fig_1.set_xlabel('Epochs[times]',fontsize=14)
    fig_1.set_ylabel('Accuracy[%]',fontsize=14)
    fig_1.legend(loc='upper left')
    fig.savefig('fig/acc/'+path_r+'_acc.eps', bbox_inches="tight", pad_inches=0.05)
    fig.savefig('fig/acc/'+path_r+'_acc.png', bbox_inches="tight", pad_inches=0.05)
    

    fig = plt.figure()
    fig_2 = fig.add_subplot(111)
    fig_2.plot(range(1, nb_epoch+1),result.history['loss'], label="training")
    fig_2.plot(result.history['val_loss'], label="validation")
    f = open('myfile.txt', 'w')
    fig_2.set_xlabel('Epochs[times]',fontsize=14)
    fig_2.set_ylabel('Loss[-]',fontsize=14)
    fig_2.legend(loc='upper left')
    fig.savefig('fig/loss/'+path_r+'_loss.eps', bbox_inches="tight", pad_inches=0.05)
    fig.savefig('fig/loss/'+path_r+'_loss.png', bbox_inches="tight", pad_inches=0.05)
    
    fig = plt.figure()
    fig_2 = fig.add_subplot(111)
    newlist = [n*100 for n in result.history['recall']]
    fig_2.plot(range(1, nb_epoch+1), newlist, label="training")
    newlist = [n*100 for n in result.history['val_recall']]
    fig_2.plot(range(1, nb_epoch+1), newlist, label="validation")
    f = open('myfile.txt', 'w')
    fig_2.set_xlabel('Epochs[times]',fontsize=14)
    fig_2.set_ylabel('Recall[%]',fontsize=14)
    fig_2.legend(loc='upper left')
    fig.savefig('fig/recall/'+path_r+'_recall.eps', bbox_inches="tight", pad_inches=0.05)
    fig.savefig('fig/recall/'+path_r+'_recall.png', bbox_inches="tight", pad_inches=0.05)

    fig = plt.figure()
    fig_2 = fig.add_subplot(111)
    newlist = [n*100 for n in result.history['precision']]
    fig_2.plot(range(1, nb_epoch+1), newlist, label="training")
    newlist = [n*100 for n in result.history['val_precision']]
    fig_2.plot(range(1, nb_epoch+1), newlist, label="validation")
    f = open('myfile.txt', 'w')
    fig_2.set_xlabel('Epochs[times]',fontsize=14)
    fig_2.set_ylabel('Precision[%]',fontsize=14)
    fig_2.legend(loc='upper left')
    fig.savefig('fig/precision/'+path_r+'_precision.eps', bbox_inches="tight", pad_inches=0.05)
    fig.savefig('fig/precision/'+path_r+'_precision.png', bbox_inches="tight", pad_inches=0.05)
    
    with open('value/'+path_r+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(result.history['accuracy'])
        writer.writerow(result.history['val_accuracy'])
        writer.writerow(result.history['loss'])
        writer.writerow(result.history['val_loss'])
        writer.writerow(result.history['recall'])
        writer.writerow(result.history['val_recall'])
        writer.writerow(result.history['precision'])
        writer.writerow(result.history['val_precision'])
    
    model.save('model/clasify_model('+path_r+').h5')

deep_learn()