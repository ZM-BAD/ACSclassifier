# ACSclassifier
该项目为本人毕业设计作品。

在本研究中，**急性冠脉综合征**(ACS)主要不良心血管事件有**出血事件**和**缺血事件**两大类，其中出血事件和缺血事件**相互独立**。

本研究通过**深度注意力机制(Attention)**来预测疾病，实际上是训练**两个二分类的分类器**。实际上对患者而言，四分类器更有意义，因为患者更加关心的是自己的出血情况和缺血情况，而一名患者一共会有四种不同的可能。但是相比于两个二分类器而言，四分类器在预测数据上会比较难看。所以本研究选择训练两个二分类器为目的。

模型架构大致如下：


![](https://raw.githubusercontent.com/ZM-BAD/ACSclassifier/master/resource/idea.jpg)



整体思路为：

1. 利用SDAE对输入层进行特征抽取
2. 利用Attention机制对抽取的feature进行加权
3. 在加权好的特征跑几个全连接层
4. 进行Dropout，防止过拟合
5. LR回归输出结果


除了算法模型以外，该研究最终还要以良好的UI呈现给用户。**最后的软件界面示意图如下**：

![](https://raw.githubusercontent.com/ZM-BAD/ACSclassifier/master/resource/panel.png)



## TODO：

- [ ] 从dataset中读取数据
- [ ] 建立SDAE，进行特征抽取
- [ ] 利用Attention机制进行注意力加权
- [ ] 跑全连接层，以及Dropout
- [ ] 为整个系统构建GUI界面
- [ ] 模块间联调