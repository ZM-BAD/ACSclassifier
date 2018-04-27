# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QFileDialog
from PyQt5.QtGui import *
from ui.new_panel import Ui_MainWindow


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)

        self.setWindowTitle("急性冠脉综合征主要不良心血管事件预测系统")
        self.setWindowIcon(QIcon("../res/PyQt5.ico"))
        self.center()
        self.choose_button.clicked.connect(self.choose_file)
        self.model_sketch.setPixmap(QPixmap("../res/label.png"))
        self.radioButton_lr.clicked.connect(self.show_lr_sketch)
        self.radioButton_sdae.clicked.connect(self.show_sdae_sketch)

    def show_lr_sketch(self):
        self.model_sketch.setPixmap(QPixmap("../res/lr_sketch.png"))

    def show_sdae_sketch(self):
        self.model_sketch.setPixmap(QPixmap("../res/sdae_sketch.png"))

    # Set the main window in the center of screen
    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def choose_file(self):
        file, ok = QFileDialog.getOpenFileName(self, "打开", "C:/", "All Files (*);; Text Files (*.txt)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())
