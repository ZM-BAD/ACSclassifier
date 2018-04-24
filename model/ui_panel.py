# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'
from tkinter import *
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk

from model.control import *

# Fonts used
ui_font = 'Microsoft YaHei UI'
small = (ui_font, 13)
big = (ui_font, 18)
text_font = 'Consolas'
text = (text_font, 13)

# Size of the ROC image
IMAGE_WIDTH = 350
IMAGE_HEIGHT = 350

# Locations of some widgets
FILEPATH_x = 255
FILEPATH_y = 30
SAMPLE_x = 165  # '样本数量'_x
SAMPLE_y = 110  # '样本数量'_y
HBA = 30  # HEIGHT_BETWEEN_'样本数量'_AND_'特征数量'
BI_x = 100  # BLEED_ROC_IMAGE_x
II_x = 550  # ISCHEMIC_ROC_IMAGE_x
RI_y = 260  # ROC_IMAGE_y

# Locations of bleed text-label
BA_x = BI_x - 10  # BLEED_AUC_x
BA_y = 640  # BLEED_AUC_y
BF1_x = 280  # BLEED_F1_SCORE_x
BF1_y = BA_y  # BLEED_F1_SCORE_y
BP_x = BA_x  # BLEED_PRECISION_x
BP_y = 680  # BLEED_PRECISION_y
BR_x = BF1_x  # BLEED_RECALL_x
BR_y = BP_y  # BLEED_RECALL_y

# Locations of ischemic text-label
IA_x = II_x - 10  # ISCHEMIC_AUC_x
IA_y = BA_y  # ISCHEMIC_AUC_x
IF1_x = 730  # ISCHEMIC_F1_SCORE_x
IF1_y = IA_y  # ISCHEMIC_F1_SCORE_y
IP_x = IA_x  # ISCHEMIC_PRECISION_x
IP_y = BP_y  # ISCHEMIC_PRECISION_y
IR_x = IF1_x  # ISCHEMIC_RECALL_x
IR_y = IP_y  # ISCHEMIC_RECALL_y

WBLAT = 90  # WIDTH_BETWEEN_LABEL_AND_TEXT


class UIPanel(object):

    def __init__(self):
        self.root = Tk()
        self.root.geometry('1000x720')
        self.root.title('急性冠脉综合征主要不良心血管事件预测系统')
        self.file_path = StringVar(value="C:/Users/ZM-BAD")  # Show the file pathname
        self._place_labels()
        self._place_text()
        self._model_selected = IntVar()
        self._model_select_buttons()
        self._place_buttons()
        self._place_roc_image()

    def _place_labels(self):
        # Select the dataset file
        Label(self.root, text='打开文件', font=small).place(x=FILEPATH_x - 80, y=FILEPATH_y, anchor=W)
        Label(self.root, textvariable=self.file_path, font=(ui_font, 12)).place(x=FILEPATH_x, y=FILEPATH_y, anchor=W)

        # Dataset info and set model param
        Label(self.root, text='数据信息', font=small).place(x=235, y=70, anchor=W)
        Label(self.root, text='模型设置', font=small).place(x=685, y=70, anchor=W)

        # Get the dataset info
        Label(self.root, text='样本数量:', font=small).place(x=SAMPLE_x, y=SAMPLE_y, anchor=W)
        Label(self.root, text='特征数量:', font=small).place(x=SAMPLE_x, y=SAMPLE_y + HBA, anchor=W)
        Label(self.root, text='出血事件:', font=small).place(x=SAMPLE_x, y=SAMPLE_y + 2 * HBA, anchor=W)
        Label(self.root, text='缺血事件:', font=small).place(x=SAMPLE_x, y=SAMPLE_y + 3 * HBA, anchor=W)

        # Set the epoch and nodes of each hidden layer(if you choose SDAE)
        # If you choose LR model, the param of SDAE hiddens will be ineffective
        Label(self.root, text='选择模型并设置参数，点击"训练"', font=small).place(x=SAMPLE_x + 395, y=SAMPLE_y, anchor=W)
        Label(self.root, text='各隐藏层节点数(以空格分隔): ', font=small).place(x=SAMPLE_x + 395, y=SAMPLE_y + 2 * HBA, anchor=W)
        Label(self.root, text='Epochs(for both 2 models)=', font=small).place(x=SAMPLE_x + 395, y=SAMPLE_y + 3 * HBA,
                                                                              anchor=W)

        # Photo titles
        Label(self.root, text="出血事件", font=big).place(x=BI_x + 125, y=RI_y - 20, anchor=W)
        Label(self.root, text="缺血事件", font=big).place(x=II_x + 125, y=RI_y - 20, anchor=W)

        # Bleed event
        Label(self.root, text="AUC: ", font=small).place(x=BA_x, y=BA_y, anchor=W)
        Label(self.root, text="F1-score: ", font=small).place(x=BF1_x, y=BF1_y, anchor=W)
        Label(self.root, text="Precision: ", font=small).place(x=BP_x, y=BP_y, anchor=W)
        Label(self.root, text="Recall: ", font=small).place(x=BR_x, y=BR_y, anchor=W)

        # Ischemic event
        Label(self.root, text="AUC: ", font=small).place(x=IA_x, y=IA_y, anchor=W)
        Label(self.root, text="F1-score: ", font=small).place(x=IF1_x, y=IF1_y, anchor=W)
        Label(self.root, text="Precision: ", font=small).place(x=IP_x, y=IP_y, anchor=W)
        Label(self.root, text="Recall: ", font=small).place(x=IR_x, y=IR_y, anchor=W)

    def _model_select_buttons(self):
        Radiobutton(self.root, text='Benchmark: LR', variable=self._model_selected, value=1, font=small).place(
            x=555, y=SAMPLE_y + 30, anchor=W)
        Radiobutton(self.root, text='研究模型: SDAE', variable=self._model_selected, value=2, font=small).place(
            x=735, y=SAMPLE_y + 30, anchor=W)

    def _place_buttons(self):
        Button(self.root, text='选择', font=(ui_font, 10), width=5, command=self._confirm_select_click).place(
            x=FILEPATH_x + 470, y=FILEPATH_y, anchor=W)

        Button(self.root, text="训练", font=(ui_font, 10), width=5, command=self._confirm_train_click).place(
            x=FILEPATH_x + 520, y=FILEPATH_y, anchor=W)

    def _place_text(self):
        self.epochs = Text(height=1, width=12, font=text)
        self.epochs.place(x=SAMPLE_x + 635, y=SAMPLE_y + 3 * HBA, anchor=W)
        self.hiddens = Text(height=1, width=12, font=text)
        self.hiddens.place(x=SAMPLE_x + 635, y=SAMPLE_y + 2 * HBA, anchor=W)

        self.bleed_auc = Text(height=1, width=10, font=text)
        self.bleed_auc.place(x=BA_x + WBLAT, y=BA_y, anchor=W)

        self.bleed_f1_score = Text(height=1, width=10, font=text)
        self.bleed_f1_score.place(x=BF1_x + WBLAT, y=BF1_y, anchor=W)

        self.bleed_precision = Text(height=1, width=10, font=text)
        self.bleed_precision.place(x=BP_x + WBLAT, y=BP_y, anchor=W)

        self.bleed_recall = Text(height=1, width=10, font=text)
        self.bleed_recall.place(x=BR_x + WBLAT, y=BR_y, anchor=W)

        self.ischemic_auc = Text(height=1, width=10, font=text)
        self.ischemic_auc.place(x=IA_x + WBLAT, y=IA_y, anchor=W)

        self.ischemic_f1_score = Text(height=1, width=10, font=text)
        self.ischemic_f1_score.place(x=IF1_x + WBLAT, y=IF1_y, anchor=W)

        self.ischemic_precision = Text(height=1, width=10, font=text)
        self.ischemic_precision.place(x=IP_x + WBLAT, y=IP_y, anchor=W)

        self.ischemic_recall = Text(height=1, width=10, font=text)
        self.ischemic_recall.place(x=IR_x + WBLAT, y=IR_y, anchor=W)

    def _place_roc_image(self):
        bleed_origin = Image.open('../res/bleed_roc.png')
        bleed_new = bleed_origin.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)

        render = ImageTk.PhotoImage(bleed_new)
        bleed = Label(self.root, image=render)
        bleed.image = render
        bleed.place(x=BI_x, y=RI_y, anchor=NW)

        ischemic_origin = Image.open('../res/ischemic_roc.png')
        ischemic_new = ischemic_origin.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(ischemic_new)
        ischemic = Label(self.root, image=render)
        ischemic.image = render
        ischemic.place(x=II_x, y=RI_y, anchor=NW)

    def _confirm_select_click(self):
        self.file_path.set(askopenfilename())
        samples, features, ischemics, bleeds = get_sample_info(self.file_path)
        Label(self.root, text=samples, font=small, foreground='SlateBlue').place(x=SAMPLE_x + 115, y=SAMPLE_y, anchor=W)
        Label(self.root, text=features, font=small, foreground='SlateBlue').place(x=SAMPLE_x + 115, y=SAMPLE_y + HBA, anchor=W)
        Label(self.root, text=bleeds, font=small, foreground='SlateBlue').place(x=SAMPLE_x + 115, y=SAMPLE_y + 2 * HBA, anchor=W)
        Label(self.root, text=ischemics, font=small, foreground='SlateBlue').place(x=SAMPLE_x + 115, y=SAMPLE_y + 3 * HBA, anchor=W)

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
