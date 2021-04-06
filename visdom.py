# # # 散点图
import torch
from visdom import Visdom
# from sklearn.datasets import load_iris
# iris_x, iris_y = load_iris(return_X_y = True)
viz = Visdom()
# viz.scatter(iris_x[:,[0, 2, 3]],
#             Y = iris_y + 1,
#             win = 'window1',env = 'main',
#             opts = dict(makersize = 1,
#                         xlabel = '*&^',
#                         ylabel = '2',
#                         zlabel = '#$%'))
# # # 折线图

# x = torch.linspace(-6, 6, 100).view(-1, 1)
# print(x)
# sigmoid = torch.nn.Sigmoid()
# sigmoidy = sigmoid(x)
# tanh = torch.nn.Tanh()
# tanhy = tanh(x)
# relu = torch.nn.ReLU()
# reluy = relu(x)
# ploty = torch.cat((sigmoidy, tanhy, reluy), dim = 1)
# plotx = torch.cat((x, x, x), dim = 1)
# viz.line(Y = ploty, X = plotx, win = 'plot line', opts = dict(legend = ['sigmoid', 'tanh', 'relu']))
import numpy as np
import math
viz.line([0.],[0.],win = 'train_loss', opts = dict(title = 'train_loss', legend = ['loss_hahaha']))
loss_list = []
for epoch in range(1000):
    loss = math.sin(epoch) / (epoch + 1)
    viz.line([loss], [epoch], win = 'train_loss', update = 'append')