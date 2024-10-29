import pandas as pd
import numpy as np
from numpy import argmax
from keras.datasets import mnist
import argparse
import matplotlib.pyplot as plt
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape((train_x.shape[0],-1))
test_x = test_x.reshape((test_x.shape[0],-1))

def loss(target_y, predict_y): #target_y: (60000,) predict_y: (60000, 10) Cross Entropy Loss
    # print(np.arange(predict_y.shape[0]), target_y)
    # print(predict_y[np.arange(predict_y.shape[0]), target_y][0])
    aranged_y = -np.log(predict_y[np.arange(predict_y.shape[0]), target_y]) #aranged_y shape: (60000,)
    return np.sum(aranged_y)

def forward(train_x, w, b): #用现有的weight来预测结果
    z = np.matmul(train_x, w) + b #(60000, 10)
    #print(z[0,:])
    # print("before softmax", np.sum(z))
    min_vals = np.min(z, axis = 1)
    min_vals = min_vals.reshape(-1, 1)
    # print(min_vals.shape)
    predict_y = np.exp(z - min_vals)/np.sum(np.exp(z - min_vals), axis = 1).reshape(-1,1)
    #print("forward predict y shape:", predict_y.shape)
    # print("after softmax",np.sum(predict_y))
    return predict_y

def backward(predict_y, train_y, train_x): #用weight的gradient来评估当前weight的表现
    aranged_train_y = np.zeros((predict_y.shape[0], 10))
    # np.arange(labels.shape[0]) 生成一个索引数组 [0, 1, ..., n_samples-1]
    # labels 本身包含每个样本的类别索引
    # 这里使用这两个数组作为行索引和列索引来将对应位置置为1
    #print("train y shape:", train_y.shape)
    # print("predict y shape:", predict_y.shape)
    aranged_train_y[np.arange(predict_y.shape[0]), train_y] = 1
    w_gradient = (np.matmul(train_x.T,(predict_y - aranged_train_y))) # w_gradient (784,10)
    # print("predict_y",predict_y)
    # print("dz",(predict_y - aranged_train_y))
    # print("sum",np.sum(np.abs(w_gradient)))
    return w_gradient, 0

def train(train_x, train_y, lr, batch_size, epoch_num):
    w = np.zeros((train_x.shape[1], 10)) #initialize weight: (784, 10)
    b = 0 #initialize bias
    iteration  = 1
    for epoch in range(epoch_num): #60000//128 = 468, 468*epoch
        i = 0
        predict_y = forward(train_x[i:i+batch_size], w, b) #predict一下y
        current_loss = loss(train_y[i:i+batch_size], predict_y)
        w_gradient, b_gradient = backward(predict_y, train_y[i:i+batch_size], train_x[i:i+batch_size]) #用predict的y和实际的y的差计算loss，然后对weight求导（因为想找到使loss最小的w）
        w = w - w_gradient * lr #更新weight的值（斜率小于0说明应该往右找）
        b = b - b_gradient * lr
        print("now is epoch:", epoch)
        while np.sum(np.abs(w_gradient * lr)) > 0.0001 or i + batch_size <= train_x.shape[0]: #不停地forward（预测），backward（评估），update（改进）
            i += batch_size
            #print("lalala:", np.sum(np.abs(w_gradient * lr)))
            predict_y = forward(train_x[i:i+batch_size], w, b)
            w_gradient, b_gradient = backward(predict_y, train_y[i:i+batch_size], train_x[i:i+batch_size])
            w = w - w_gradient * lr
            b = b - b_gradient * lr
            iteration += 1
            current_loss = loss(train_y[i:i+batch_size], predict_y)
            #print("iteration:",iteration, "loss:", current_loss)
    return w, b

def convert_array_to_one_hot(numbers, num_classes): #Not in use
    one_hot_matrix = np.zeros((len(numbers), num_classes))
    one_hot_matrix[np.arange(len(numbers)), numbers] = 1
    return one_hot_matrix

def test(test_x, test_y, w, b):
    test_outcome = forward(test_x, w, b)
    max_indices = np.argmax(test_outcome, axis = 1)
    diff = (max_indices == test_y)
    sum_diff = np.sum(diff)
    print(sum_diff/test_x.shape[0])
    return sum_diff/test_x.shape[0]

if __name__ == "__main__":
    w, b = train(train_x[:]/256.0, train_y[:], lr = 0.0001, batch_size = 60000, epoch_num = 50)
    test(test_x/256.0, test_y, w, b) #epoch = 50: 0.9175(32, 64, 128), 0.9176(1, 8, 16, 256), 09174(512)

    matrix = w  

    #设置图像大小
    plt.figure(figsize=(10, 5))

    # 遍历每一列，将其转换为 28x28 的图像并绘制
    for i in range(10):
        ax = plt.subplot(2, 5, i + 1)
        img = matrix[:, i].reshape(28, 28)  # 将每一列重塑为 28x28 的图像
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # 关闭坐标轴

    plt.show()
