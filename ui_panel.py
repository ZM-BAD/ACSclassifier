# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'
from control import *
from tkinter import *
from tkinter.filedialog import askopenfilename

ui_font = 'Microsoft YaHei UI'
text_font = 'Consolas'


class UIPanel(object):

    def __init__(self):
        self.root = Tk()
        self.root.geometry('1000x600')
        self.root.title('急性冠脉综合征主要不良心血管事件预测系统')
        self.file_path = StringVar(value="C:/Users/ZM-BAD")  # 显示选择的文件的路径
        self._place_text()
        self._place_labels()
        self._place_buttons()

    def _place_labels(self):
        Label(self.root, textvariable=self.file_path, font=(ui_font, 12)).place(x=200, y=40, anchor=W)
        Label(self.root, text='打开', font=(ui_font, 15)).place(x=150, y=40, anchor=W)
        Label(self.root, text="F1-score: ", font=(ui_font, 18)).place(x=80, y=550, anchor=W)
        Label(self.root, text="Precision: ", font=(ui_font, 18)).place(x=380, y=550, anchor=W)
        Label(self.root, text="Recall: ", font=(ui_font, 18)).place(x=700, y=550, anchor=W)

    def _place_buttons(self):
        Button(self.root, text='选择', font=(ui_font, 10), width=5,
               command=lambda: self.file_path.set(askopenfilename())).place(x=750, y=40, anchor=W)

        Button(self.root, text="确定", font=(ui_font, 10), width=5,
               command=self._confirm_click).place(x=800, y=40, anchor=W)
        Button(self.root, text="清零", font=(ui_font, 10), width=5,
               command=self._clear_click).place(x=700, y=40, anchor=W)

    def _place_text(self):
        # 以下分别为F1-score, recall, precision三个参数
        self.f1_score = Text(height=1, width=10, font=(text_font, 18))
        self.f1_score.place(x=200, y=550, anchor=W)

        self.precision = Text(height=1, width=10, font=(text_font, 18))
        self.precision.place(x=510, y=550, anchor=W)

        self.recall = Text(height=1, width=10, font=(text_font, 18))
        self.recall.place(x=790, y=550, anchor=W)

    def _confirm_click(self):
        # 首先将三个text里面原有的内容都清零，不管原来有没有
        self.f1_score.delete(1.0, END)
        self.precision.delete(1.0, END)
        self.recall.delete(1.0, END)
        f1_score, precision, recall = calc_numerical_result(self.file_path)
        self.f1_score.insert(END, f1_score)
        self.precision.insert(END, precision)
        self.recall.insert(END, recall)

    def _clear_click(self):
        self.f1_score.delete(1.0, END)
        self.precision.delete(1.0, END)
        self.recall.delete(1.0, END)

    def show(self):
        self.root.mainloop()


if __name__ == "__main__":
    ui = UIPanel()
    ui.show()
