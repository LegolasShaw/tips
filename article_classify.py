# coding: utf-8
# @author: Shaw
# @datetime: 2019-02-25 15:14
# @Name: article_classify.py

import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

LABEL_MAP = {'体育': 1, '女性': 2, '文学': 3, '校园': 4}

with open('./text_classification/stop/stopword.txt', encoding='utf-8') as f:
    STOP_WORDS = [line.strip() for line in f.readlines()]


def load_article(base_path, documents=[], labels=[]):
    for file_names in os.listdir(base_path):
        child_path = os.path.join(base_path, file_names)
        if not os.path.isdir(child_path):
            label = base_path.split('\\')[-1]
            labels.append(label)
            child_path = child_path.replace('\\', '/')
            with open(child_path, 'rb') as f:
                content = f.read()
                word_list = list(jieba.cut(content))
                documents.append(' '.join(word_list))
        else:
            load_article(base_path=child_path, documents=documents, labels=labels)


def load_train_func(train_document, train_label, test_document, test_label):
    tf = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.5)  # 词向量
    train_features = tf.fit_transform(train_document)
    clf = MultinomialNB(alpha=0.001).fit(train_features, train_label)

    test_tf = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.5, vocabulary=tf.vocabulary_)
    test_features = test_tf.fit_transform(test_document)
    predicted_labels = clf.predict(test_features)

    x = metrics.accuracy_score(test_label, predicted_labels)
    return x


if __name__ == "__main__":
    train_documents = list()
    train_labels = list()
    test_document = list()
    test_label = list()
    load_article(base_path='./text_classification/train', documents=train_documents, labels=train_labels)
    load_article(base_path='./text_classification/test', documents=test_document, labels=test_label)
    for i in range(len(train_labels)):
        train_labels[i] = LABEL_MAP[train_labels[i]]

    for i in range(len(test_label)):
        test_label[i] = LABEL_MAP[test_label[i]]
    print(load_train_func(train_document=train_documents, train_label=train_labels, test_document=test_document, test_label=test_label))

