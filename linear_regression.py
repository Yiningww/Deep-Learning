import pandas as pd
import numpy as np
from keras.datasets import mnist
import argparse
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape((train_x.shape[0],-1))
test_x = test_x.reshape((test_x.shape[0],-1))

def loss(target_y, predict_y):
    n = target_y.shape[0]
    return 1.0/float(n) * np.sum((target_y - predict_y)**2)

def forward(train_x, w, b): #用现有的weight来预测结果
    predict_y = np.matmul(train_x, w) + b
    #print(predict_y.shape)
    return predict_y

def backward(predict_y, train_y, train_x): #用weight的gradient来评估当前weight的表现
    n = predict_y.shape[0]
    w_gradient = -2.0/float(n) * np.matmul(train_x.T, (train_y - predict_y))
    b_gradient = -2.0/float(n) * np.sum(train_y - predict_y)
    # print(b_gradient)
    return w_gradient, b_gradient

def train(train_x, train_y, lr):
    w = np.zeros((train_x.shape[1])) #initialize weight (784,)
    b = 0 #initialize bias
    predict_y = forward(train_x, w, b) #predict一下y
    w_gradient, b_gradient = backward(predict_y, train_y, train_x) #用predict的y和实际的y的差计算loss，然后对weight求导（因为想找到使loss最小的w）
    w = w - w_gradient * lr #更新weight的值（斜率小于0说明应该往右找）
    b = b - b_gradient * lr
    iteration  = 1
    current_loss = loss(train_y, predict_y)
    print(iteration, current_loss)
    while iteration  <= 100 and np.sum(np.abs(w_gradient * lr)) > 0.01: #不停地forward（预测），backward（评估），update（改进）
        predict_y = forward(train_x, w, b)
        w_gradient, b_gradient = backward(predict_y, train_y, train_x)
        w = w - w_gradient * lr
        b = b - b_gradient * lr
        iteration += 1
        current_loss = loss(train_y, predict_y)
        print(iteration, current_loss)
    return w, b

def test(test_x, test_y, w, b):
    test_outcome = np.round(forward(test_x, w, b))
    print(test_outcome)
    diff = (test_outcome == test_y)
    print(diff)
    sum_diff = np.sum(diff)
    print(sum_diff/test_y.shape[0])
    return sum_diff/test_y.shape[0]


if __name__ == "__main__":
    w, b = train(train_x[:]/256.0, train_y[:], 0.01)
    test(test_x/256.0, test_y, w, b)

