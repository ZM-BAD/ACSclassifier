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
        self.root.geometry('1000x720')
        self.root.title('急性冠脉综合征主要不良心血管事件预测系统')
        self.file_path = StringVar(value="C:/Users/ZM-BAD")  # Show the file pathname
        self._place_text()
        self._place_labels()
        self._place_buttons()
        self._place_roc_image()

    def _place_labels(self):
        Label(self.root, textvariable=self.file_path, font=(ui_font, 12)).place(x=200, y=40, anchor=W)
        Label(self.root, text='打开', font=(ui_font, 13)).place(x=130, y=40, anchor=W)
        Label(self.root, text="出血事件: ", font=(ui_font, 20)).place(x=200, y=90, anchor=W)
        Label(self.root, text="缺血事件: ", font=(ui_font, 20)).place(x=700, y=90, anchor=W)

        # 出血事件
        Label(self.root, text="F1-score: ", font=(ui_font, 13)).place(x=150, y=570, anchor=W)
        Label(self.root, text="Precision: ", font=(ui_font, 13)).place(x=150, y=620, anchor=W)
        Label(self.root, text="Recall: ", font=(ui_font, 13)).place(x=150, y=670, anchor=W)
        # 缺血事件
        Label(self.root, text="F1-score: ", font=(ui_font, 13)).place(x=650, y=570, anchor=W)
        Label(self.root, text="Precision: ", font=(ui_font, 13)).place(x=650, y=620, anchor=W)
        Label(self.root, text="Recall: ", font=(ui_font, 13)).place(x=650, y=670, anchor=W)

    def _place_buttons(self):
        Button(self.root, text='选择', font=(ui_font, 10), width=5,
               command=lambda: self.file_path.set(askopenfilename())).place(x=750, y=40, anchor=W)

        Button(self.root, text="确定", font=(ui_font, 10), width=5,
               command=self._confirm_click).place(x=800, y=40, anchor=W)

    def _place_text(self):
        self.bleed_f1_score = Text(height=1, width=10, font=(text_font, 13))
        self.bleed_f1_score.place(x=250, y=570, anchor=W)

        self.bleed_precision = Text(height=1, width=10, font=(text_font, 13))
        self.bleed_precision.place(x=250, y=620, anchor=W)

        self.bleed_recall = Text(height=1, width=10, font=(text_font, 13))
        self.bleed_recall.place(x=250, y=670, anchor=W)

        self.ischemic_f1_score = Text(height=1, width=10, font=(text_font, 13))
        self.ischemic_f1_score.place(x=750, y=570, anchor=W)

        self.ischemic_precision = Text(height=1, width=10, font=(text_font, 13))
        self.ischemic_precision.place(x=750, y=620, anchor=W)

        self.ischemic_recall = Text(height=1, width=10, font=(text_font, 13))
        self.ischemic_recall.place(x=750, y=670, anchor=W)

    def _place_roc_image(self):
        bleed_origin = Image.open('../resource/bleed_roc.png')
        bleed_new = bleed_origin.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT), Image.ANTIALIAS)

        render = ImageTk.PhotoImage(bleed_new)
        bleed = Label(self.root, image=render)
        bleed.image = render
        bleed.place(x=50, y=130, anchor=NW)

        ischemic_origin = Image.open('../resource/ischemic_roc.png')
        ischemic_new = ischemic_origin.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(ischemic_new)
        ischemic = Label(self.root, image=render)
        ischemic.image = render
        ischemic.place(x=550, y=130, anchor=NW)

    def _confirm_click(self):
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
