import tkinter as tk


class UIPanel(object):
    def __init__(self):
        self.top = tk.Tk()
        self.top.geometry("1000x600")

    def show(self):
        self.top.mainloop()


if __name__ == '__main__':
    ui = UIPanel()
    ui.show()
