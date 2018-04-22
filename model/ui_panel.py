# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'
from tkinter import *
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk

from model.control import *

ui_font = 'Microsoft YaHei UI'
text_font = 'Consolas'
IMAGE_WIDTH = 350
IMAGE_HEIGHT = 350


class UIPanel(object):

    def __init__(self):
        self.root = Tk()
        self.root.geometry('800x720')
        self.root.title('急性冠脉综合征主要不良心血管事件预测系统')
        self.file_path = StringVar(value="C:/Users/ZM-BAD")  # Show the file pathname
        self._place_labels()
        self._place_text()

        self._place_buttons()
        self._place_roc_image()

    def _place_labels(self):
        Label(self.root, text='打开文件', font=(ui_font, 13)).place(x=90, y=30, anchor=W)
        Label(self.root, textvariable=self.file_path, font=(ui_font, 12)).place(x=170, y=30, anchor=W)

        Label(self.root, text='数据信息', font=(ui_font, 13)).place(x=150, y=70, anchor=W)
        Label(self.root, text='模型设置', font=(ui_font, 13)).place(x=550, y=70, anchor=W)

        Label(self.root, text='样本数量:', font=(ui_font, 13)).place(x=60, y=110, anchor=W)
        Label(self.root, text='特征数量:', font=(ui_font, 13)).place(x=220, y=110, anchor=W)
        Label(self.root, text='出血事件:', font=(ui_font, 13)).place(x=220, y=140, anchor=W)
        Label(self.root, text='缺血事件:', font=(ui_font, 13)).place(x=60, y=140, anchor=W)

        Label(self.root, text="出血事件: ", font=(ui_font, 18)).place(x=150, y=190, anchor=W)
        Label(self.root, text="缺血事件: ", font=(ui_font, 18)).place(x=550, y=190, anchor=W)

        # 出血事件
        Label(self.root, text="F1-score: ", font=(ui_font, 13)).place(x=120, y=570, anchor=W)
        Label(self.root, text="Precision: ", font=(ui_font, 13)).place(x=120, y=620, anchor=W)
        Label(self.root, text="Recall: ", font=(ui_font, 13)).place(x=120, y=670, anchor=W)
        # 缺血事件
        Label(self.root, text="F1-score: ", font=(ui_font, 13)).place(x=620, y=570, anchor=W)
        Label(self.root, text="Precision: ", font=(ui_font, 13)).place(x=620, y=620, anchor=W)
        Label(self.root, text="Recall: ", font=(ui_font, 13)).place(x=620, y=670, anchor=W)

    def _place_buttons(self):
        Button(self.root, text='选择', font=(ui_font, 10), width=5,
               command=self._confirm_select_click).place(x=620, y=30, anchor=W)

        Button(self.root, text="训练", font=(ui_font, 10), width=5,
               command=self._confirm_train_click).place(x=670, y=30, anchor=W)

    def _place_text(self):
        self.bleed_f1_score = Text(height=1, width=15, font=(text_font, 13))
        self.bleed_f1_score.place(x=230, y=570, anchor=W)

        self.bleed_precision = Text(height=1, width=15, font=(text_font, 13))
        self.bleed_precision.place(x=230, y=620, anchor=W)

        self.bleed_recall = Text(height=1, width=15, font=(text_font, 13))
        self.bleed_recall.place(x=230, y=670, anchor=W)

        self.ischemic_f1_score = Text(height=1, width=15, font=(text_font, 13))
        self.ischemic_f1_score.place(x=730, y=570, anchor=W)

        self.ischemic_precision = Text(height=1, width=15, font=(text_font, 13))
        self.ischemic_precision.place(x=730, y=620, anchor=W)

        self.ischemic_recall = Text(height=1, width=15, font=(text_font, 13))
        self.ischemic_recall.place(x=730, y=670, anchor=W)

    def _place_roc_image(self):
        bleed_origin = Image.open('../res/bleed_roc.png')
        bleed_new = bleed_origin.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)

        render = ImageTk.PhotoImage(bleed_new)
        bleed = Label(self.root, image=render)
        bleed.image = render
        bleed.place(x=25, y=230, anchor=NW)

        ischemic_origin = Image.open('../res/ischemic_roc.png')
        ischemic_new = ischemic_origin.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(ischemic_new)
        ischemic = Label(self.root, image=render)
        ischemic.image = render
        ischemic.place(x=425, y=230, anchor=NW)

    def _confirm_select_click(self):
        self.file_path.set(askopenfilename())
        samples, features, ischemics, bleeds = get_sample_info(self.file_path)
        Label(self.root, text=samples, font=(ui_font, 13)).place(x=140, y=110, anchor=W)
        Label(self.root, text=features, font=(ui_font, 13)).place(x=300, y=110, anchor=W)
        Label(self.root, text=bleeds, font=(ui_font, 13)).place(x=300, y=140, anchor=W)
        Label(self.root, text=ischemics, font=(ui_font, 13)).place(x=140, y=140, anchor=W)

    def _confirm_train_click(self):
        # Delete origin values in text-box
        self.bleed_f1_score.delete(1.0, END)
        self.bleed_precision.delete(1.0, END)
        self.bleed_recall.delete(1.0, END)
        self.ischemic_f1_score.delete(1.0, END)
        self.ischemic_precision.delete(1.0, END)
        self.ischemic_recall.delete(1.0, END)

        # Get new values and write it
        bleed_f1_score, bleed_precision, bleed_recall = calc_numerical_result(self.file_path)
        ischemic_f1_score, ischemic_precision, ischemic_recall = calc_numerical_result(self.file_path)
        self.bleed_f1_score.insert(END, bleed_f1_score)
        self.bleed_precision.insert(END, bleed_precision)
        self.bleed_recall.insert(END, bleed_recall)
        self.ischemic_f1_score.insert(END, ischemic_f1_score)
        self.ischemic_precision.insert(END, ischemic_precision)
        self.ischemic_recall.insert(END, ischemic_recall)

    def show(self):
        self.root.mainloop()


if __name__ == "__main__":
    ui = UIPanel()
    ui.show()
