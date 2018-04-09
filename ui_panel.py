from tkinter import *
from tkinter.filedialog import askopenfilename

import _thread
import sys


class UIPanel(object):
    def __init__(self):
        self.root = Tk()
        self.root.geometry('1000x600')
        self.root.title('急性冠脉综合征主要不良心血管事件预测系统')

        # 创建模型选择的Radio button
        self._model_selected = IntVar()  # 代表被选择的模型

        self.out_file_str = StringVar(value="select file")  # 输出文件路径
        self._place_some_text()  # 面板中是存放text的白底框
        self.place_labels()  # '文件浏览'，训练样本数量等标签，一共7个
        self.place_buttons()  # 选择，预测这俩按钮
        self._change_stdout()

    def place_labels(self):
        Label(self.root, textvariable=self.out_file_str, justify=LEFT).place(x=20, y=250, anchor=W)
        Label(textvariable=StringVar(value="epochs"), justify=LEFT).place(x=20, y=90, anchor=W)
        Label(textvariable=StringVar(value="output_n_epochs"), justify=LEFT).place(x=20, y=130, anchor=W)
        Label(textvariable=StringVar(value="lstm_size"), justify=LEFT).place(x=20, y=170, anchor=W)

        Label(textvariable=StringVar(value="precision"), justify=LEFT).place(x=150, y=160, anchor=W)
        Label(textvariable=StringVar(value="recall"), justify=LEFT).place(x=150, y=190, anchor=W)
        Label(textvariable=StringVar(value="f_score"), justify=LEFT).place(x=150, y=220, anchor=W)

        file_browser = Label(self.root, text='文件浏览', bg='LightSkyBlue', font=('Microsoft YaHei', 19))
        file_browser.pack()

    def _place_some_text(self):

        self.epochs_text = Text(self.root, height=1, width=10)
        self.epochs_text.insert(END, 1000)
        self.epochs_text.place(x=20, y=110, anchor=W)

        self.output_n_epoch_text = Text(self.root, height=1, width=10)
        self.output_n_epoch_text.insert(END, 20)
        self.output_n_epoch_text.place(x=20, y=150, anchor=W)

        self.lstm_size_text = Text(self.root, height=1, width=10)
        self.lstm_size_text.insert(END, 200)
        self.lstm_size_text.place(x=20, y=190, anchor=W)

        # 以下分别为F1-score, recall, precision三个参数
        self.precision_text = Text(height=2, width=40)
        self.precision_text.place(x=220, y=160, anchor=W)

        self.recall_text = Text(height=2, width=40)
        self.recall_text.place(x=220, y=190, anchor=W)

        self.f1_score_text = Text(height=2, width=40)
        self.f1_score_text.place(x=220, y=220, anchor=W)

    def place_buttons(self):
        Button(self.root, text='选择', command=lambda: self.out_file_str.set(askopenfilename())).place(x=20, y=280,
                                                                                                     anchor=W)
        Button(self.root, text="确定", command=self._confirm_click).place(x=120, y=280, anchor=W)

    def _confirm_click(self):
        self.auc_text.delete('1.0', END)
        self.precision_text.delete('1.0', END)
        self.recall_text.delete('1.0', END)
        self.f1_score_text.delete('1.0', END)
        self.loss_out_text.delete('1.0', END)

        value = self._model_selected.get()
        result_file = self.out_file_str.get()

        if result_file is None:
            return
        if value == 0:
            print("select a model")
        elif value == 1:
            print("waiting....")
            _thread.start_new_thread(self._call_basic_lstm_experiment, (result_file,))
        elif value == 2:
            print('waiting....')
            _thread.start_new_thread(self._call_bi_lstm_experiment, (result_file,))
        else:
            pass

    def _change_stdout(self):
        sys.stdout.write = self.__write_to_loss_text

    def __write_to_loss_text(self, *args):
        self.loss_out_text.config(state=NORMAL)
        self.loss_out_text.insert(END, args[0])
        self.loss_out_text.see(END)
        self.loss_out_text.config(state=DISABLED)

    def show(self):
        self.root.mainloop()


if __name__ == "__main__":
    ui = UIPanel()
    ui.show()
