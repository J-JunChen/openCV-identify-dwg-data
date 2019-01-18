import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QLabel
# QMessageBox, QDockWidget, QListWidget

from Ui_host_computer import Ui_MainWindow

# image_label = 0


class mywindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('铺砖机器人--上位机')

        self.choose_pushButton.clicked.connect(self.choose_pushButton_clicked)

    def choose_pushButton_clicked(self):
        """  
            选择CAD文件按钮事件
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileRoad, _ = QFileDialog.getOpenFileName(
            self,
            "选择CAD文件",
            "",
            "JPG Files (*.jpg);;CAD Files (*.dwg)",
            options=options)
        if fileRoad:
            fileName = os.path.split(fileRoad)[1]  #分离文件名
            self.cad_label.setText(fileName)  #修改label 的text
            self.load_image(fileRoad)

    def load_image(self, image):
        """  
            加载图片：
                1、每次load_image，就清空原来的label.clear()
                2、setPixmap(pixmap.scaled(size(),KeepAsceptRatio))表示按图像比例显示
        """
        # image_label = QLabel(self.image_label)

        self.image_label.clear() #每次选择
        pixmap = QPixmap(image)
       
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(),
                          QtCore.Qt.KeepAspectRatio))  #radio：根据图像比例显示图片
        # self.image_label.setScaledC
    
    
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())