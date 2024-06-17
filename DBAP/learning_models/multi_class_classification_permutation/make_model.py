from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, BatchNormalization
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import csv
import time

def load_data(path_r, ship_num):
    x = []
    y = []
    path = "learn_data/"+path_r+".csv"
    data_num = sum([1 for _ in open(path)])
    with open(path) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        text = [[int(v) for v in row] for row in l]
    for index in range(data_num):
        x.append(text[index][:-1])
    for index in range(data_num):
        y.append(np.eye(ship_num + 1)[text[index][-1]])
    return x, y

def build_model(ship_num, berth_num):
    layyer_1 = 3 #3-4?
    input_berth = []
    x1 = []
    dense = []
    for index in range(layyer_1):
        dense.append(Dense(4, activation='relu'))
    for index_1 in range(berth_num):
        input_berth.append(Input(shape = (2,)))
        x1.append(dense[0](input_berth[index_1]))
        for index_2 in range(layyer_1 - 1):
            x1[index_1] = dense[index_2 + 1](x1[index_1])

    input_ship = []
    x2 = []
    dense = []
    for index in range(layyer_1):
        dense.append(Dense(30, activation = 'relu'))
        
    for index_1 in range(ship_num):
        input_ship.append(Input(shape = (3 + berth_num,)))
        x2.append(dense[0](input_ship[index_1]))
        for index_2 in range(layyer_1 - 1):
            x2[index_1] = dense[index_2 + 1](x2[index_1])

    input_cond = Input(shape = (4,))
    x3 = Dense(3 * 4, activation='relu')(input_cond)
    for index in range(layyer_1 - 1):
        x3 = Dense(3 * 4, activation='relu')(x3)
    
    z = concatenate(x1 + x2 + [x3], axis=-1)

    layyer_2 = 4#2-4?
    #layyer_2 = 5
    #z = Dense(ship_num * 3 + berth_num * 2 +berth_num * ship_num + 3, activation = "relu")(z)
    #z = Dense(2*(ship_num * 3 + berth_num * 2 +berth_num * ship_num + 3), activation = "relu")(z)
    #z = Dense(200, activation = "relu")(z)
    #z = Dense(124, activation = "relu")(z)
    z = Dense(100, activation = "relu")(z)
    #z = Dense(ship_num * 5, activation = "relu")(z)
    z = BatchNormalization()(z)
    for index in range(layyer_2-1):
        #z = Dense(128, activation = "relu")(z)
        #z = Dense(ship_num * 3 + berth_num * 2 +berth_num * ship_num + 3, activation = "relu")(z)
        #z = Dense(ship_num * 3, activation = "relu")(z)
        z = Dense(100, activation = "relu")(z)
        #z = Dense(ship_num * 5, activation = "relu")(z)
    z = Dense(ship_num + 1, activation = "softmax")(z)
    model = Model(inputs = input_berth + input_ship + [input_cond], outputs = z)
    learning_rate = 0.001
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer = adam,
                  metrics=['accuracy'])
    return model

def deep_learn():
    ship_num = 60
    berth_num = 13
    
    path_r="train(3000)"
    x_train , y_train = load_data(path_r, ship_num)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    input_berth = []
    for index in range(berth_num):
        x, x_train = np.split(x_train, [2,], 1)
        input_berth.append(x)
    
    input_ship =[]
    for index in range(ship_num):
        x, x_train = np.split(x_train, [3 + berth_num, ], 1)
        input_ship.append(x)
        
    input_cond = x_train
    x_train = input_berth + input_ship + [input_cond]
    
    
    # path_r="validate5"
    # x_val , y_val = load_data(path_r, ship_num)
    # x_val = np.array(x_val)
    # y_val = np.array(y_val)
    
    # input_berth = []
    # for index in range(berth_num):
    #     x, x_val = np.split(x_val,[2,],1)
    #     input_berth.append(x)
    
    # input_ship =[]
    # for index in range(ship_num):
    #     x, x_val = np.split(x_val,[3 + berth_num,],1)
    #     input_ship.append(x)
    
    # input_cond = x_val
    # x_val = input_berth + input_ship + [input_cond]

    model = build_model(ship_num, berth_num)
    nb_epoch = 100
    batch_size = 514
    batch_size = 1028
    
    # result = model.fit(x_train,
    #                     y_train,
    #                     batch_size = batch_size,
    #                     epochs = nb_epoch,
    #                     verbose = 1,
    #                     validation_data = (x_val, y_val)
    #                     )
    
    t = time.time()
    
    result = model.fit(x_train,
                        y_train,
                        batch_size = batch_size,
                        epochs = nb_epoch,
                        verbose = 1,
                        validation_split = 0.2)
    
    print(time.time() - t)
    
    # dt_now = datetime.datetime.now()
    # path_r = path_r + str(dt_now)
    path_r = "model"
    
    # plt.rcParams['font.family'] = 'Times New Roman'
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
    
    # fig.savefig('fig/'+path_r+'_acc.eps', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    # fig.savefig('fig/'+path_r+'_acc.svg', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    # fig.savefig('fig/'+path_r+'_acc.png', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    # fig.savefig('fig/'+path_r+'_acc.jpg', bbox_inches="tight", pad_inches=0.05, dpi = 300)
    # fig.savefig('fig/'+path_r+'_acc.eps', bbox_inches="tight", pad_inches=0.05)
    # fig.savefig('fig/'+path_r+'_acc.png', bbox_inches="tight", pad_inches=0.05)
    fig = plt.figure()
    fig_2 = fig.add_subplot(111)
    fig_2.plot(range(1, nb_epoch+1),result.history['loss'], label="training")
    fig_2.plot(result.history['val_loss'], label="validation")
    # f = open('myfile.txt', 'w')
    fig_2.set_xlabel('Epochs[times]',fontsize=14)
    fig_2.set_ylabel('Loss[-]',fontsize=14)
    fig_2.legend(loc=0)
    
    # fig.savefig('fig/'+path_r+'_loss.eps', bbox_inches = "tight", pad_inches = 0.05, dpi = 300)
    # fig.savefig('fig/'+path_r+'_loss.svg', bbox_inches = "tight", pad_inches = 0.05, dpi = 300)
    # fig.savefig('fig/'+path_r+'_loss.png', bbox_inches = "tight", pad_inches = 0.05, dpi = 300)
    # fig.savefig('fig/'+path_r+'_loss.jpg', bbox_inches = "tight", pad_inches = 0.05, dpi = 300)
    # fig.savefig('fig/'+path_r+'_loss.eps', bbox_inches="tight", pad_inches=0.05)
    # fig.savefig('fig/'+path_r+'_loss.png', bbox_inches="tight", pad_inches=0.05)
    # with open('fig_value/'+path_r+'.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(result.history['accuracy'])
    #     writer.writerow(result.history['val_accuracy'])
    #     writer.writerow(result.history['loss'])
    #     writer.writerow(result.history['val_loss'])
    # model.save('model/clasify_model('+path_r+').h5')
    
deep_learn()
