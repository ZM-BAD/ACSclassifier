# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from ui.panel import Ui_MainWindow
from model.control import *


class ModelThread(QThread):
    finished = pyqtSignal()

    def __init__(self, dataset_path, model, epochs, hiddens):
        """
        :param model: "lr" or "sdae"
        :param epochs: epochs
        :param hiddens: may be used
        """
        super(ModelThread, self).__init__()
        self.dataset_path = dataset_path
        self.model = model
        self.epochs = epochs
        self.hiddens = hiddens

    def run(self):
        if self.model == "lr":
            lr_experiment(self.dataset_path, self.epochs)
        else:
            sdae_experiment(self.dataset_path, self.epochs, self.hiddens)

        self.finished.emit()


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)
        # Show the main window
        self.setWindowTitle("急性冠脉综合征主要不良心血管事件预测系统")
        self.setWindowIcon(QIcon("../res/pics/PyQt5.ico"))
        self.setWindowFlags(Qt.MSWindowsFixedSizeDialogHint)
        self.center()
        self.dataset_is_selected = False

        # Place some labels
        self.model_sketch.setPixmap(QPixmap("../res/pics/choose_model.png"))
        self.sample_statistics.setPixmap(QPixmap("../res/pics/blank_statistics.png"))
        self.loss_curve.setPixmap(QPixmap("../res/pics/blank_loss.png"))
        self.label_bleeding_event_pic.setPixmap(QPixmap("../res/pics/blank_result.png"))
        self.label_ischemic_event_pic.setPixmap(QPixmap("../res/pics/blank_result.png"))
        self.label_lr.setStyleSheet("QLabel { background-color : HotPink }")
        self.label_sdae.setStyleSheet("QLabel { background-color : CornflowerBlue }")

        self.choose_button.clicked.connect(self.choose_file)
        self.confirm_button.clicked.connect(self.confirm)
        self.train_button.clicked.connect(self.train)
        self.clear_button.clicked.connect(self.clear)
        self.radioButton_lr.clicked.connect(self.show_lr_sketch)
        self.radioButton_sdae.clicked.connect(self.show_sdae_sketch)

        self.lr_thread = ModelThread(self.file_dir.text(), "lr", self.epochs.text(), hiddens=None)
        self.sdae_thread = ModelThread(self.file_dir.text(), "sdae", self.epochs.text(), hiddens=None)

        # Once the training thread is finished, the train_button should be accessible
        self.lr_thread.finished.connect(self.thread_finished)
        self.sdae_thread.finished.connect(self.thread_finished)

    # Show the LR/SDAE model sketch
    def show_lr_sketch(self):
        self.model_sketch.setPixmap(QPixmap("../res/pics/lr_sketch.png"))

    def show_sdae_sketch(self):
        self.model_sketch.setPixmap(QPixmap("../res/pics/sdae_sketch.png"))

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
            self.invalid_dataset()
        else:
            self.dataset_is_selected = True
            draw_sample_info_statistics(path)
            self.sample_statistics.setPixmap(QPixmap("../res/venn.png"))
            os.remove("../res/venn.png")

    # Train the model
    def train(self):
        # Check if the dataset is selected
        if not self.dataset_is_selected:
            self.dataset_is_not_selected()
            return

        # Check if the model is selected
        if not self.radioButton_lr.isChecked() and not self.radioButton_sdae.isChecked():
            self.no_model_chosen()
            return

        # Check if the epoch is valid
        epoch = self.epochs.text()
        if len(epoch) == 0 or not epoch.isdigit():
            self.invalid_epoch()
            return

        # Train LR model
        if self.radioButton_lr.isChecked():
            # First, set the button inaccessible
            self.train_button.setEnabled(False)
            # Then, set the parameters for the model
            self.lr_thread.dataset_path = self.file_dir.text()
            self.lr_thread.epochs = epoch
            # Start the thread
            self.lr_thread.start()
            self.loss_curve.setPixmap(QPixmap("../res/pics/waiting.png"))
            self.label_bleeding_event_pic.setPixmap(QPixmap("../res/pics/waiting.png"))
            self.label_ischemic_event_pic.setPixmap(QPixmap("../res/pics/waiting.png"))

        # Train SDAE model
        if self.radioButton_sdae.isChecked():
            if len(self.hiddens.text()) == 0:
                self.invalid_sdae_hiddens()
                return
            else:
                hiddens = self.hiddens.text().split(' ')
                valid = True
                for i in hiddens:
                    if not i.isdigit():
                        valid = False
                        break
                if not valid:
                    self.invalid_sdae_hiddens()
                    return
                else:
                    self.train_button.setEnabled(False)
                    self.sdae_thread.dataset_path = self.file_dir.text()
                    self.sdae_thread.epochs = epoch
                    self.sdae_thread.hiddens = hiddens
                    self.sdae_thread.start()
                    self.loss_curve.setPixmap(QPixmap("../res/pics/waiting.png"))
                    self.label_bleeding_event_pic.setPixmap(QPixmap("../res/pics/waiting.png"))
                    self.label_ischemic_event_pic.setPixmap(QPixmap("../res/pics/waiting.png"))

    # Clear existing results
    def clear(self):
        self.loss_curve.setPixmap(QPixmap("../res/pics/blank_loss.png").scaled(400, 335))
        self.label_bleeding_event_pic.setPixmap(QPixmap("../res/pics/blank_result.png"))
        self.label_ischemic_event_pic.setPixmap(QPixmap("../res/pics/blank_result.png"))
        self.epochs.clear()
        self.hiddens.clear()

    # Set the train_button accessible
    def thread_finished(self):
        self.loss_curve.setPixmap(QPixmap("../res/output/loss_curve.png").scaled(400, 335))
        self.label_ischemic_event_pic.setPixmap(QPixmap("../res/output/ischemic.png"))
        self.label_bleeding_event_pic.setPixmap(QPixmap("../res/output/bleeding.png"))
        self.train_button.setEnabled(True)

    # Some exceptions and solutions
    def dataset_is_not_selected(self):
        reply = QMessageBox.warning(self, "错误", "未选择数据集", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        print(reply)

    def invalid_dataset(self):
        reply = QMessageBox.warning(self, "错误", "请选择正确的数据集", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        print(reply)

    def no_model_chosen(self):
        reply = QMessageBox.warning(self, "错误", "没有选择模型", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        print(reply)

    def invalid_epoch(self):
        reply = QMessageBox.warning(self, "错误", "无效的epoch参数\n请重新输入", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        print(reply)

    def invalid_sdae_hiddens(self):
        reply = QMessageBox.warning(self, "错误", "无效的隐藏层输入\n请重新输入", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        print(reply)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())
