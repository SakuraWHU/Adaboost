# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_moons
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import gif


class AdaboostDemo:

    def __init__(self, X, y, val_X=None, val_y=None, learning_rate=1., method=''):
        # plt的图窗
        self.fig = plt.figure(1)
        self.ax = plt.gca()
        """
        输入的X为N*2矩阵, y为一维向量, y的值只能取1或-1
        :param X: 数据点
        :param y: 数据点标记
        """
        self.X = X
        self.y = y
        self.val_X = val_X
        self.val_y = val_y
        self.method = method
        # 给每个弱分类器一个衰减, 避免过拟合
        self.learning_rate = learning_rate
        # 样本的个数
        self.num_samples = len(self.X)
        # 初始化数据样本的权重为1/N
        self.sample_weight = np.full(self.num_samples, 1.0 / self.num_samples)
        # python list用来存储所有的弱分类器对象
        self.classifiers = []
        # 储存在每一步的错误率
        self.errors_list = []
        # 定义弱分类器, 这里我们直接调用sklearn的决策树, max_depth=1代表着这是一个一层决策树, 也就是决策树桩
        self.alphas = []
        # metrics
        self.score = []
        # val_metrics
        self.val_score = []

    def predict(self, data=None, labels=None):
        """
        预测数据点的分类
        :param labels: 数据标签
        :param data: 数据
        """
        if data is None:
            data = self.X
            labels = self.y
        # 计算弱分类器线性加权组合的结果
        predictions = np.zeros([len(data)]).astype("float")
        for classifier, alpha in zip(self.classifiers, self.alphas):
            predictions += alpha * classifier.predict(data)
        # 对结果取符号
        predictions = np.sign(predictions)
        # 获取ac
        if labels is not None:
            ac = f1_score(predictions, labels, average='macro')
            return predictions, ac
        else:
            return predictions

    def contour_plot(self, data=None, labels=None, interval=0.1, title="adaboost", test=False, validation=False):
        """
        Adaboost可视化
        :param labels: 数据label
        :param data: 数据点
        :param interval: 等高线图网格的间隔
        :param title: 等高线图的标题
        """
        if data is None:
            data = self.X
            labels = self.y
        if labels is None:
            labels = np.ones([len(data)])
        # 获取网格
        x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
        y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, interval), np.arange(y_min, y_max, interval))
        # 将网格的X, Y轴拼接用来进行等高线的计算
        X_grid = np.concatenate([np.expand_dims(np.ravel(xx), axis=-1),
                                 np.expand_dims(np.ravel(yy), axis=-1)], axis=-1)
        # X_grid的形状[batch(数据点数量), 2]
        # 计算分类边界(等高线)
        Z_grid = self.predict(data=X_grid)
        Z_grid = Z_grid.reshape(xx.shape)

        plt.cla()
        # 等高线
        self.ax.contourf(xx, yy, Z_grid, alpha=.8, cmap=plt.cm.BrBG)
        # 散点
        self.ax.scatter(data[:, 0], data[:, 1], c=labels,
                        cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolors='k')
        self.ax.set_title(title)
        f = plt.gcf()
        if test == False and validation == False:
            f.savefig('pic/' + self.method + '/' + str(len(self.classifiers)) + '.jpg')
        elif test == True and validation == False:
            f.savefig('pic/' + self.method + '/test.jpg')
        elif test == False and validation == True:
            f.savefig('pic/' + self.method + '/validation.jpg')

    def __next__(self, plot=True):
        # 定义弱分类器
        if self.method == 'DecisionTree_1':
            classifier = DecisionTreeClassifier(max_depth=1)
        elif self.method == 'DecisionTree_2':
            classifier = DecisionTreeClassifier(
                max_depth=2, min_samples_split=20,
                min_samples_leaf=5)
        elif self.method == 'SVM':
            classifier = SVC()
        elif self.method == 'ExtraTree':
            classifier = ExtraTreeClassifier()
        elif self.method == 'RandomForest':
            classifier = RandomForestClassifier()
        elif self.method == 'LogisticRegression':
            classifier = LogisticRegression()
        elif self.method == 'GradientBoosting':
            classifier = GradientBoostingClassifier()
        elif self.method == 'Perceptron':
            classifier = Perceptron()
        elif self.method == 'Bayesian':
            classifier = GaussianNB()
        else:
            classifier = None

        # 用弱分类器拟合数据
        classifier.fit(self.X, self.y, sample_weight=self.sample_weight)
        # 得到弱分类器对数据的推断, 也就是h(x)
        predictions = classifier.predict(self.X)
        # 计算错误率epsilon
        error_rate = np.average((predictions != self.y), weights=self.sample_weight) + 1e-5
        self.errors_list.append(error_rate)
        # 计算alpha
        alpha = self.learning_rate * (np.log((1 - error_rate) / error_rate)) / 2
        # 计算t+1的权重
        self.sample_weight *= np.exp(-alpha * self.y * predictions)
        # 归一化, 归一化因子为Z: sum(self.sample_weight)
        self.sample_weight /= np.sum(self.sample_weight)
        # 记录当前弱分类器对象
        self.classifiers.append(classifier)
        # 记录当前弱分类器权重
        self.alphas.append(alpha)
        # 计算accuracy_score
        _, ac = self.predict()
        self.score.append(ac)
        if self.val_X is not None and self.val_y is not None:
            _, ac_val = self.predict(self.val_X, self.val_y)
            self.val_score.append(ac_val)
        # 画图
        if plot:
            return self.contour_plot(
                title=model.method + " adaboost step " + str(len(model.classifiers)) + "\n f1_score is: {:.2f}".format(
                    ac))
        else:
            return ac


if __name__ == '__main__':
    raw = pd.read_csv("TrainSet.txt", header=0, names=['x1', 'x2', 'label'])
    Train_set = raw.values
    X_train = Train_set[::, :2]
    y_train = Train_set[::, 2]

    raw = pd.read_csv("TestSet.txt", header=0, names=['x1', 'x2', 'label'])
    Train_set = raw.values
    X_test = Train_set[::, :2]
    y_test = Train_set[::, 2]

    # X, y = make_moons(n_samples=1500, noise=0.3, random_state=4321)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.333, random_state=1234)
    # 生成的样本label是0,1 而adaboost中label是-1,1
    y_train[np.where(y_train == 0)] = -1
    y_test[np.where(y_test == 0)] = -1
    # y_val[np.where(y_val == 0)] = -1

    # ---------------------------------------------------------------------#
    # ----下面的代码中,method参数需要改动,range中的数字代表弱分类器个数------------#
    # ---------------------------------------------------------------------#
    # 调用adaboost类
    # 方法有 'DecisionTree_1' 'DecisionTree_2' 'SVM' 'ExtraTree'
    #       'RandomForest' 'LogisticRegression' 'GradientBoosting' 'Perceptron' 'Bayesian'
    #
    model = AdaboostDemo(X_train, y_train,
                         method='DecisionTree_2')

    for i in range(50):
        model.__next__(plot=True)
    # ---------------------------------------------------------------------#
    # ----改动上面的代码中,下面的代码用于出图与生成gif---------------------------#
    # ---------------------------------------------------------------------#
    model.contour_plot(
        title=model.method + " adaboost step " + str(len(model.classifiers)) + "\n f1_score is: {:.2f}".format(
            model.predict()[1]), test=False, validation=False)

    image_list = ["pic/" + model.method + '/' + str(x) + ".jpg" for x in range(1, len(model.classifiers) + 1)]
    gif_name = "pic/" + model.method + '/' + model.method + '_gif.gif'
    gif.create_gif(image_list, gif_name)
    model.contour_plot(X_test, y_test,
                       title=model.method + " adaboost\ntest classifiers:" + str(
                           len(model.classifiers)) + "\n f1_score is: {:.2f}".format(
                           model.predict(X_test, y_test)[1]), test=True, validation=False)
    # model.contour_plot(X_val, y_val,
    #                    title=model.method + " adaboost\nvalidation classifiers:" + str(
    #                        len(model.classifiers)) + "\nvalidation f1_score is: {:.2f}".format(
    #                        model.predict(X_val, y_val)[1]), test=False, validation=True)
    f = plt.gcf()
    plt.cla()
    plt.plot(range(1, len(model.classifiers) + 1), model.errors_list)
    plt.title(f'model final error_rate: {model.errors_list[-1]:.2f}')
    f.savefig('pic/' + model.method + '/error_rate.jpg')
    plt.cla()
    plt.plot(range(1, len(model.classifiers) + 1), model.score)
    plt.title(f'training final f1_score: {model.score[-1]:.2f}')
    f.savefig('pic/' + model.method + '/score.jpg')
    plt.cla()
    # plt.plot(range(1, len(model.classifiers) + 1), model.val_score)
    # plt.title(f'validation final f1_score: {model.val_score[-1]:.2f}')
    # f.savefig('pic/' + model.method + '/val_score.jpg')
