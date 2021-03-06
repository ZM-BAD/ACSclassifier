# ACSclassifier

## Introduction
该项目为本人毕业设计作品。

在本研究中，**急性冠脉综合征**(ACS)主要不良心血管事件包括**出血事件**和**缺血事件**两大类，其中出血事件和缺血事件**相互独立**。

本研究通过**SDAE**对患者特征进行抽取，根据抽取的高阶特征来预测疾病。本研究本质上是训练**两个二分类的分类器**。

**本研究中使用到的数据集不对外提供，还请见谅。代码中数据集路径为/res/dataset.csv**

模型架构大致如下：

<div align=center><img src="https://raw.githubusercontent.com/ZM-BAD/ACSclassifier/master/res/docs/new_model.png"/></div>


整体思路为：  
1. 利用SDAE对输入层进行特征抽取
2. 运用LR(逻辑回归)进行分类，作为benchmark
3. 在抽取的特征基础之上跑Softmax分类
4. 输出F1-score、Recall、Precision、AUC值


除了算法模型以外，该研究最终还要以良好的UI呈现给用户。**软件界面如下**：


![](https://raw.githubusercontent.com/ZM-BAD/ACSclassifier/master/res/docs/panel_1.png)

![](https://raw.githubusercontent.com/ZM-BAD/ACSclassifier/master/res/docs/panel_2.png)

![](https://raw.githubusercontent.com/ZM-BAD/ACSclassifier/master/res/docs/panel_3.png)


## Dependencies

Programming Language: Python 3.6.5(64-bit)  
Machine Learning Framework: TensorFlow ~~1.7.0~~ 1.8.0  
GUI lib: ~~Tkinter~~ PyQt 5.10.1



## Install Requirements

```
pip install -r requirements.txt
```



## Getting Started

1. Clone this repository
2. Open this project by PyCharm
3. Generate panel.py file from panel.ui by PyUIC5
4. Run 'call_panel'




## TODO：

- [x] 从dataset中读取数据
- [x] 建立SDAE，进行特征抽取
- [x] 运用LR进行分类，作为benchmark，得出AUC、F1-score、Recall、Precision等模型指标
- [x] 利用Softmax对抽取的特征进行分类
- [x] 为整个系统构建GUI界面
- [x] 模块间联调，整个代码跑起来
- [x] Code Review
- [x] 调试参数，获得多数数据进行对照分析
