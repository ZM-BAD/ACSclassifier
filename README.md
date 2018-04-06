# ACSclassifier
该项目为本人毕业设计作品。

急性冠脉综合征(ACS)主要不良心血管事件有**出血事件**和**缺血事件**两大类，其中出血事件和缺血事件相互独立。


本研究通过深度注意力机制来预测疾病，模型架构大致如下：


![easonjim](https://raw.githubusercontent.com/ZM-BAD/ACSclassifier/master/idea.jpg)



整体思路为：

1. 利用SDAE对输入层进行特征抽取
2. 利用Attention机制对抽取的feature进行加权
3. 在加权好的特征上构建MLP，然后跑几个全连接层
4. 进行Dropout，防止过拟合
5. LR回归输出结果

## TODO：

- [ ] 从dataset中读取数据
- [ ] 建立SDAE，进行特征抽取
- [ ] 利用Attention机制进行注意力加权
- [ ] 跑全连接层，以及Dropout
- [ ] 为整个系统构建GUI界面