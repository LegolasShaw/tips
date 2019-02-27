# !/usr/bin/env python
# -*- coding: utf-8 -*-

# SOAP 的调用方式版本最好使用python 2.7

from suds.client import Client  # 导入suds.client 模块下的Client类

wsdl_url = "http://172.31.86.226:8080/wsdl/ICrew_Task"


def say_hello_test(url):
    client = Client(url)  # 创建一个webservice接口对象
    client.service.write_physical_examination_info(10, 0, 'guoxk', '2019-02-22 00:00:00', '2019-02-24 00:00:00', 'ckg', 'ckg', 'test')
    req = str(client.last_sent())  # 保存请求报文，因为返回的是一个实例，所以要转换成str
    response = str(client.last_received())  # 保存返回报文，返回的也是一个实例
    print(req)  # 打印请求报文
    print(response)


if __name__ == '__main__':
    say_hello_test(wsdl_url)