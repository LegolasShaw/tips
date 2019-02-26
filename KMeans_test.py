# coding: utf-8
# @author: Shaw
# @datetime: 2019-02-26 13:15
# @Name: KMeans_test.py

# KMeans 包含在 sklearn.cluster
from sklearn.cluster import KMeans

# KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')

# n_clusters 即K的值  max_iter 最大的迭代次数
# n_init：初始化中心点的运算次数，默认是 10。程序是否快速收敛和中心点的选择关系非常大
# init 即初始值得选择 默认的方式采用优化过的 k-means ++ 方式
# algorithm : k-meansde 实现算法 有 "auto", "full", "elkan" 三种方式

from sklearn import preprocessing
import pandas as pd
import PIL.Image as image
import numpy as np

if __name__ == "__main1__":
    data = pd.read_csv('./kmeans/data.csv', encoding="gbk")

    feature_col = ['2019年国际排名', '2018世界杯', '2015亚洲杯']
    train_x = data[feature_col]
    df = pd.DataFrame(train_x)
    kmeans = KMeans(n_clusters=3)
    min_max = preprocessing.MinMaxScaler()
    train_x = min_max.fit_transform(train_x)
    kmeans.fit(train_x)
    predit_y = kmeans.predict(train_x)

    result = pd.concat((data, pd.DataFrame(predit_y)), axis=1)
    result.rename({0: u'聚类'}, axis=1, inplace=True)

if __name__ == "__main__":
    def load_data(file_path):
        with open(file_path, 'rb') as f:
            # 读取文件
            data =[]
            # 得到文件像素
            img = image.open(f)
            # 得到 图像尺寸
            width, height = img.size

            for x in range(width):
                for y in range(height):
                    c1, c2, c3 = img.getpixel((x, y))
                    data.append([c1, c2, c3])

        min_max = preprocessing.MinMaxScaler()
        data = min_max.fit_transform(data)
        return np.mat(data), width, height

    img, width, height = load_data('./kmeans/weixin.jpg')

    # 用K-Means 对图像进行2 聚类
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(img)
    label = kmeans.predict(img)

    label = label.reshape([width, height])

    pic_make = image.new("L", (width, height))
    for x in range(width):
        for y in range(height):
            pic_make.putpixel((x, y), int(256/(label[x][y]+1))-1)

    pic_make.save("weixin_reshape.jpg", "JPEG")





