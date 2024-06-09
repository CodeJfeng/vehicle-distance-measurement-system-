import math

import matplotlib.pyplot as plt
import numpy as np


# # 定义函数
# def f(x):
#     return x ** 1.7 *0.012
#
#
# plt.rcParams['font.sans-serif']=['SimHei']
# # 创建数据点
# x = range(0, 120)  # 从-10到10的整数
# y = [f(i) for i in x]  # 对应每个x值的y值
#
# # 绘制图像
# plt.plot(x, y)
# plt.title('速度与刹车距离的关系')  # 图像标题
# plt.xlabel('速度（km/h）')  # x轴标签
# plt.ylabel('刹车距离（m）')  # y轴标签
# plt.show()  # 显示图像

# def frelu(x):
#     return  0.1*x if x < 0 else x
#
# def silu(x):
#     return x / (1 + math.exp(-x))
#
# plt.rcParams['font.sans-serif']=['SimHei']
#
# x = np.arange(-10, 10.01, 0.01)
# y = [frelu(i) for i in x]
# plt.plot(x,y , label='LeakyRELU')
#
# y2 = [silu(i) for  i in x]
# plt.plot(x,y2 , label='SiLU')
#
# plt.legend()
# plt.show()
#


list = [['汽车', '辆', 17.7],['汽车', '辆', 18.7]]
list.sort(key= lambda x: x[2] )
print(list)