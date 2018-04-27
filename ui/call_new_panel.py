# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QFileDialog
from PyQt5.QtGui import *
from ui.new_panel import Ui_MainWindow
from model.control import *


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)
        # Show the main window
        self.setWindowTitle("急性冠脉综合征主要不良心血管事件预测系统")
        self.setWindowIcon(QIcon("../res/PyQt5.ico"))
        self.center()

        # Place some labels
        self.model_sketch.setPixmap(QPixmap("../res/model_choose_label.png"))
        self.sample_statistics.setPixmap(QPixmap("../res/label.png").scaled(400, 360))
        self.loss_curve.setPixmap(QPixmap("../res/label.png").scaled(400, 360))
        self.label_bleeding_event_pic.setPixmap(QPixmap("../res/label.png"))
        self.label_ischemic_event_pic.setPixmap(QPixmap("../res/label.png"))
        self.label_lr.setStyleSheet("QLabel { background-color : HotPink }")
        self.label_sdae.setStyleSheet("QLabel { background-color : CornflowerBlue }")

        self.choose_button.clicked.connect(self.choose_file)
        self.train_button.clicked.connect(self.train)
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
        file = QFileDialog.getOpenFileName(self, "打开", self.file_dir.text(), "All Files (*)")
        file_path = file[0]
        self.file_dir.setText(file_path)
        draw_sample_info_statistics(file_path)
        self.sample_statistics.setPixmap(QPixmap("../res/venn.png"))

    def train(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())
