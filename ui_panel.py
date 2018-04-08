from tkinter import *
from tkinter.filedialog import askopenfilename


class UIPanel(object):
    def __init__(self):
        self.top = Tk()
        self.top.geometry("1000x600")
        self.out_file_str = StringVar(value="select file")

        self.select_file_and_predict()

    def show(self):
        self.top.mainloop()

    # 选择数据文件并预测
    def select_file_and_predict(self):
        Button(self.top, text='选择', command=lambda: self.out_file_str.set(askopenfilename())).place(x=20, y=280,
                                                                                                    anchor=W)


if __name__ == '__main__':
    ui = UIPanel()
    ui.show()
