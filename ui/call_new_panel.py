# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import sys
import os
from PyQt5.QtWidgets import *
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
        self.confirm_button.clicked.connect(self.confirm)
        self.train_button.clicked.connect(self.train)
        self.radioButton_lr.clicked.connect(self.show_lr_sketch)
        self.radioButton_sdae.clicked.connect(self.show_sdae_sketch)

    # Show the LR/SDAE model sketch
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

    def confirm(self):
        path = self.file_dir.text()
        if os.path.abspath(path) != os.path.abspath('../res/dataset.csv'):
            self.invalid_file()
        else:
            draw_sample_info_statistics(path)
            self.sample_statistics.setPixmap(QPixmap("../res/venn.png"))
            os.remove("../res/venn.png")

    # Train the model
    def train(self):
        # Check if the model is selected
        if not self.radioButton_lr.isChecked() and not self.radioButton_sdae.isChecked():
            self.no_model_chosen()
            return

        # Check if the epoch is valid
        epoch = self.epochs.text()
        if len(epoch) == 0 or int(epoch) == 0:
            self.invalid_epoch()
            return

        # Train LR model
        if self.radioButton_lr.isChecked():
            lr_experiment(self.file_dir.text(), epoch)

        # Train SDAE model
        if self.radioButton_sdae.isChecked():
            if len(self.hiddens.text()) == 0:
                self.invalid_sdae()
                return
            else:
                hiddens = self.hiddens.text().split(' ')
                valid = True
                for i in hiddens:
                    if not i.isdigit():
                        valid = False
                        break
                if not valid:
                    self.invalid_sdae()
                    return
                else:
                    sdae_experiment(self.file_dir.text(), epoch, hiddens)

    def invalid_file(self):
        reply = QMessageBox.warning(self, "错误", "请选择正确的数据集", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        print(reply)

    def no_model_chosen(self):
        reply = QMessageBox.warning(self, "错误", "没有选择模型", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        print(reply)

    def invalid_epoch(self):
        reply = QMessageBox.warning(self, "错误", "无效的epoch参数\n请重新输入", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        print(reply)

    def invalid_sdae(self):
        reply = QMessageBox.warning(self, "错误", "无效的隐藏层输入\n请重新输入", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        print(reply)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())
