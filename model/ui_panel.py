# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'
from tkinter import *
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk

from model.control import *

ui_font = 'Microsoft YaHei UI'
text_font = 'Consolas'
DEFAULT_WIDTH = 400
DEFAULT_HEIGHT = 400


class UIPanel(object):

    def __init__(self):
        self.root = Tk()
        self.root.geometry('1000x600')
        self.root.title('急性冠脉综合征主要不良心血管事件预测系统')
        self.file_path = StringVar(value="C:/Users/ZM-BAD")  # 显示选择的文件的路径
        self._place_text()
        self._place_labels()
        self._place_buttons()
        self._place_roc_image()

    def _place_labels(self):
        Label(self.root, textvariable=self.file_path, font=(ui_font, 12)).place(x=200, y=40, anchor=W)
        Label(self.root, text='打开', font=(ui_font, 15)).place(x=150, y=40, anchor=W)
        Label(self.root, text="F1-score: ", font=(ui_font, 18)).place(x=80, y=550, anchor=W)
        Label(self.root, text="Precision: ", font=(ui_font, 18)).place(x=380, y=550, anchor=W)
        Label(self.root, text="Recall: ", font=(ui_font, 18)).place(x=700, y=550, anchor=W)
        Label(self.root, text="样本数量: ", font=(ui_font, 20)).place(x=100, y=200, anchor=W)
        Label(self.root, text="测试数量: ", font=(ui_font, 20)).place(x=100, y=300, anchor=W)
        Label(self.root, text="训练时长: ", font=(ui_font, 20)).place(x=100, y=400, anchor=W)

    def _place_buttons(self):
        Button(self.root, text='选择', font=(ui_font, 10), width=5,
               command=lambda: self.file_path.set(askopenfilename())).place(x=750, y=40, anchor=W)

        Button(self.root, text="确定", font=(ui_font, 10), width=5,
               command=self._confirm_click).place(x=800, y=40, anchor=W)

    def _place_text(self):
        # 以下分别为F1-score, recall, precision三个参数
        self.f1_score = Text(height=1, width=10, font=(text_font, 18))
        self.f1_score.place(x=200, y=550, anchor=W)

        self.precision = Text(height=1, width=10, font=(text_font, 18))
        self.precision.place(x=510, y=550, anchor=W)

        self.recall = Text(height=1, width=10, font=(text_font, 18))
        self.recall.place(x=790, y=550, anchor=W)

        self.train_num = Text(height=1, width=10, font=(text_font, 18))
        self.train_num.place(x=250, y=200, anchor=W)

        self.test_num = Text(height=1, width=10, font=(text_font, 18))
        self.test_num.place(x=250, y=300, anchor=W)

        self.time = Text(height=1, width=10, font=(text_font, 18))
        self.time.place(x=250, y=400, anchor=W)

    def _place_roc_image(self):
        origin_image = Image.open('./resource/ROC-curve.png')
        # print(origin_image.width)
        # print(origin_image.height)
        # 对图片大小进行标准化处理
        new_image = origin_image.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT), Image.ANTIALIAS)

        render = ImageTk.PhotoImage(new_image)
        roc_curve = Label(self.root, image=render)
        roc_curve.image = render
        roc_curve.place(x=500, y=80, anchor=NW)

    def _confirm_click(self):
        # 首先将text里面原有的内容都清零，不管原来有没有
        self.f1_score.delete(1.0, END)
        self.precision.delete(1.0, END)
        self.recall.delete(1.0, END)
        self.train_num.delete(1.0, END)
        self.test_num.delete(1.0, END)
        self.time.delete(1.0, END)

        # 得到新的值，然后写进去
        train_num, test_num, time = get_sample_info(self.file_path)
        f1_score, precision, recall = calc_numerical_result(self.file_path)
        self.train_num.insert(END, train_num)  # 训练样本数量
        self.test_num.insert(END, test_num)  # 测试数量
        self.time.insert(END, time)  # 训练耗时
        self.f1_score.insert(END, f1_score)
        self.precision.insert(END, precision)
        self.recall.insert(END, recall)

    def show(self):
        self.root.mainloop()


if __name__ == "__main__":
    ui = UIPanel()
    ui.show()
