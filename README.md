# ACSclassifier

## Introduction
该项目为本人毕业设计作品。

在本研究中，**急性冠脉综合征**(ACS)主要不良心血管事件有**出血事件**和**缺血事件**两大类，其中出血事件和缺血事件**相互独立**。

本研究通过**SDAE**对患者特征进行抽取，根据抽取的高阶特征来预测疾病。本研究本质上是训练**两个二分类的分类器**。不过对患者而言，四分类器更有意义，因为患者更加关心的是自己的出血情况和缺血情况，而一名患者一共会有四种不同的可能。但是相比于两个二分类器而言，四分类器在预测数据上会比较难看。所以本研究选择训练两个二分类器为目的。

模型架构大致如下：

![](https://raw.githubusercontent.com/ZM-BAD/ACSclassifier/master/res/new_model.png)


~~但是实际可能不按照模型来实现，所以模型仅为参考~~  
~~毕业设计要赶DDL，我能怎么办，我也很绝望啊~~  

整体思路为：  
1. 利用SDAE对输入层进行特征抽取
2. 运用朴素的LR(逻辑回归)进行分类，作为benchmark
3. 在抽取的特征基础之上跑Softmax分类
4. 进行Dropout，防止过拟合
5. 输出F1-score、Recall、Precision、AUC值


除了算法模型以外，该研究最终还要以良好的UI呈现给用户。经过迭代开发，**第一版软件界面示意图如下**：

![](https://raw.githubusercontent.com/ZM-BAD/ACSclassifier/master/res/panel.png)

## 技术路线
编程语言：Python 3.6.5(64-bit)  
机器学习框架：TensorFlow 1.7  
GUI Lib: ~~Tkinter~~ PyQt 5.10.1

## TODO：

- [x] 从dataset中读取数据
- [x] 建立SDAE，进行特征抽取
- [ ] 运用LR进行分类，作为benchmark，得出AUC、F1-score、Recall、Precision等模型指标
- [ ] 利用Softmax对抽取的特征进行分类
- [x] 为整个系统构建GUI界面
- [ ] 模块间联调
