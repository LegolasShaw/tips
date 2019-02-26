# CART 分类树 算法

# coding: utf-8

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 准备数据
digits = load_digits()

# 获取特征集合分类标示
features = digits.data
labels = digits.target

# 抽取 训练数据集 和 测试数据集
train_feature, test_feature, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)

# 创建CART 分类树
clf = DecisionTreeClassifier(criterion='gini')

# 拟合 构造CART 分类树
tree_model = clf.fit(train_feature, train_labels)

# 分类树的预测
test_predict = tree_model.predict(test_feature)


# 预测结果与测试集结果作比对
score = accuracy_score(test_labels, test_predict)
print("CART 分类树准确率 %.4lf" % score)