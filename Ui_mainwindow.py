# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\WangHeng\Documents\PyQt_code\multi_window\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1013, 719)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_image = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox_image.setObjectName("groupBox_image")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_image)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(0, 0, 0, 1)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_image)
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.image_4 = QtWidgets.QLabel(self.groupBox_4)
        self.image_4.setText("")
        self.image_4.setPixmap(QtGui.QPixmap(":/img/newblack.jpg"))
        self.image_4.setObjectName("image_4")
        self.verticalLayout_8.addWidget(self.image_4, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.label_4 = QtWidgets.QLabel(self.groupBox_4)
        self.label_4.setMinimumSize(QtCore.QSize(0, 50))
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_8.addWidget(self.label_4)
        self.gridLayout.addWidget(self.groupBox_4, 1, 3, 1, 1)
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox_image)
        self.groupBox_8.setTitle("")
        self.groupBox_8.setObjectName("groupBox_8")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.groupBox_8)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.image_8 = QtWidgets.QLabel(self.groupBox_8)
        self.image_8.setText("")
        self.image_8.setPixmap(QtGui.QPixmap(":/img/newblack.jpg"))
        self.image_8.setObjectName("image_8")
        self.verticalLayout_12.addWidget(self.image_8, 0, QtCore.Qt.AlignHCenter)
        self.label_8 = QtWidgets.QLabel(self.groupBox_8)
        self.label_8.setMinimumSize(QtCore.QSize(0, 50))
        self.label_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_12.addWidget(self.label_8)
        self.gridLayout.addWidget(self.groupBox_8, 3, 3, 1, 1)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_image)
        self.groupBox_6.setTitle("")
        self.groupBox_6.setObjectName("groupBox_6")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.groupBox_6)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.image_6 = QtWidgets.QLabel(self.groupBox_6)
        self.image_6.setText("")
        self.image_6.setPixmap(QtGui.QPixmap(":/img/newblack.jpg"))
        self.image_6.setObjectName("image_6")
        self.verticalLayout_10.addWidget(self.image_6, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.label_6 = QtWidgets.QLabel(self.groupBox_6)
        self.label_6.setMinimumSize(QtCore.QSize(0, 50))
        self.label_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_10.addWidget(self.label_6)
        self.gridLayout.addWidget(self.groupBox_6, 3, 1, 1, 1)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_image)
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.image_5 = QtWidgets.QLabel(self.groupBox_5)
        self.image_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.image_5.setText("")
        self.image_5.setPixmap(QtGui.QPixmap(":/img/newblack.jpg"))
        self.image_5.setAlignment(QtCore.Qt.AlignCenter)
        self.image_5.setObjectName("image_5")
        self.verticalLayout_9.addWidget(self.image_5)
        self.label_5 = QtWidgets.QLabel(self.groupBox_5)
        self.label_5.setMinimumSize(QtCore.QSize(0, 50))
        self.label_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_5.setWordWrap(True)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_9.addWidget(self.label_5)
        self.gridLayout.addWidget(self.groupBox_5, 3, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_image)
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.image_3 = QtWidgets.QLabel(self.groupBox_3)
        self.image_3.setText("")
        self.image_3.setPixmap(QtGui.QPixmap(":/img/newblack.jpg"))
        self.image_3.setObjectName("image_3")
        self.verticalLayout_7.addWidget(self.image_3, 0, QtCore.Qt.AlignHCenter)
        self.label_3 = QtWidgets.QLabel(self.groupBox_3)
        self.label_3.setMinimumSize(QtCore.QSize(0, 50))
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_7.addWidget(self.label_3)
        self.gridLayout.addWidget(self.groupBox_3, 1, 2, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_last = QtWidgets.QPushButton(self.groupBox_image)
        self.pushButton_last.setObjectName("pushButton_last")
        self.horizontalLayout.addWidget(self.pushButton_last)
        self.pushButton_generate = QtWidgets.QPushButton(self.groupBox_image)
        self.pushButton_generate.setObjectName("pushButton_generate")
        self.horizontalLayout.addWidget(self.pushButton_generate)
        self.pushButton_next = QtWidgets.QPushButton(self.groupBox_image)
        self.pushButton_next.setObjectName("pushButton_next")
        self.horizontalLayout.addWidget(self.pushButton_next)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pushButton_autostart = QtWidgets.QPushButton(self.groupBox_image)
        self.pushButton_autostart.setObjectName("pushButton_autostart")
        self.horizontalLayout_4.addWidget(self.pushButton_autostart)
        self.pushButton_autostop = QtWidgets.QPushButton(self.groupBox_image)
        self.pushButton_autostop.setObjectName("pushButton_autostop")
        self.horizontalLayout_4.addWidget(self.pushButton_autostop)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.gridLayout.addLayout(self.verticalLayout_3, 5, 1, 1, 2)
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_image)
        self.groupBox_7.setTitle("")
        self.groupBox_7.setObjectName("groupBox_7")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.groupBox_7)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.image_7 = QtWidgets.QLabel(self.groupBox_7)
        self.image_7.setText("")
        self.image_7.setPixmap(QtGui.QPixmap(":/img/newblack.jpg"))
        self.image_7.setObjectName("image_7")
        self.verticalLayout_11.addWidget(self.image_7, 0, QtCore.Qt.AlignHCenter)
        self.label_7 = QtWidgets.QLabel(self.groupBox_7)
        self.label_7.setMinimumSize(QtCore.QSize(0, 50))
        self.label_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_7.setWordWrap(True)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_11.addWidget(self.label_7)
        self.gridLayout.addWidget(self.groupBox_7, 3, 2, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_image)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.image_1 = QtWidgets.QLabel(self.groupBox)
        self.image_1.setText("")
        self.image_1.setPixmap(QtGui.QPixmap(":/img/newblack.jpg"))
        self.image_1.setAlignment(QtCore.Qt.AlignCenter)
        self.image_1.setObjectName("image_1")
        self.verticalLayout_5.addWidget(self.image_1)
        self.label_1 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_1.sizePolicy().hasHeightForWidth())
        self.label_1.setSizePolicy(sizePolicy)
        self.label_1.setMinimumSize(QtCore.QSize(219, 50))
        self.label_1.setTextFormat(QtCore.Qt.AutoText)
        self.label_1.setScaledContents(False)
        self.label_1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_1.setWordWrap(True)
        self.label_1.setObjectName("label_1")
        self.verticalLayout_5.addWidget(self.label_1)
        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_image)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.image_2 = QtWidgets.QLabel(self.groupBox_2)
        self.image_2.setText("")
        self.image_2.setPixmap(QtGui.QPixmap(":/img/newblack.jpg"))
        self.image_2.setAlignment(QtCore.Qt.AlignCenter)
        self.image_2.setObjectName("image_2")
        self.verticalLayout_6.addWidget(self.image_2, 0, QtCore.Qt.AlignHCenter)
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setMinimumSize(QtCore.QSize(0, 50))
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_6.addWidget(self.label_2)
        self.gridLayout.addWidget(self.groupBox_2, 1, 1, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout)
        self.verticalLayout.addWidget(self.groupBox_image)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1013, 23))
        self.menuBar.setObjectName("menuBar")
        self.menu = QtWidgets.QMenu(self.menuBar)
        self.menu.setObjectName("menu")
        self.actionLoadImage = QtWidgets.QMenu(self.menu)
        self.actionLoadImage.setObjectName("actionLoadImage")
        self.helpmenu = QtWidgets.QMenu(self.menuBar)
        self.helpmenu.setObjectName("helpmenu")
        MainWindow.setMenuBar(self.menuBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actionLoadModel = QtWidgets.QAction(MainWindow)
        self.actionLoadModel.setObjectName("actionLoadModel")
        self.actionFromFile = QtWidgets.QAction(MainWindow)
        self.actionFromFile.setObjectName("actionFromFile")
        self.actionFromCamera = QtWidgets.QAction(MainWindow)
        self.actionFromCamera.setObjectName("actionFromCamera")
        self.actionSaveCaptions = QtWidgets.QAction(MainWindow)
        self.actionSaveCaptions.setObjectName("actionSaveCaptions")
        self.action_Exit = QtWidgets.QAction(MainWindow)
        self.action_Exit.setObjectName("action_Exit")
        self.actionInstruction = QtWidgets.QAction(MainWindow)
        self.actionInstruction.setObjectName("actionInstruction")
        self.actionLoadVideo = QtWidgets.QAction(MainWindow)
        self.actionLoadVideo.setObjectName("actionLoadVideo")
        self.actionFromVideo = QtWidgets.QAction(MainWindow)
        self.actionFromVideo.setObjectName("actionFromVideo")
        self.actionLoadImage.addAction(self.actionFromFile)
        self.actionLoadImage.addAction(self.actionFromVideo)
        self.actionLoadImage.addAction(self.actionFromCamera)
        self.menu.addAction(self.actionLoadModel)
        self.menu.addAction(self.actionLoadImage.menuAction())
        self.menu.addAction(self.actionSaveCaptions)
        self.menu.addAction(self.action_Exit)
        self.helpmenu.addAction(self.actionInstruction)
        self.menuBar.addAction(self.menu.menuAction())
        self.menuBar.addAction(self.helpmenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_image.setTitle(_translate("MainWindow", "图像信息解译"))
        self.label_4.setText(_translate("MainWindow", "4."))
        self.label_8.setText(_translate("MainWindow", "8."))
        self.label_6.setText(_translate("MainWindow", "6."))
        self.label_5.setText(_translate("MainWindow", "5."))
        self.label_3.setText(_translate("MainWindow", "3."))
        self.pushButton_last.setText(_translate("MainWindow", "上一批图像"))
        self.pushButton_generate.setText(_translate("MainWindow", "生成语句"))
        self.pushButton_next.setText(_translate("MainWindow", "下一批图像"))
        self.pushButton_autostart.setText(_translate("MainWindow", "开始自动解译"))
        self.pushButton_autostop.setText(_translate("MainWindow", "停止自动解译"))
        self.label_7.setText(_translate("MainWindow", "7."))
        self.label_1.setText(_translate("MainWindow", "1."))
        self.label_2.setText(_translate("MainWindow", "2."))
        self.menu.setTitle(_translate("MainWindow", "文件(&F)"))
        self.actionLoadImage.setTitle(_translate("MainWindow", "输入数据"))
        self.helpmenu.setTitle(_translate("MainWindow", "帮助(&H)"))
        self.actionLoadModel.setText(_translate("MainWindow", "网络模型"))
        self.actionFromFile.setText(_translate("MainWindow", "输入图像"))
        self.actionFromCamera.setText(_translate("MainWindow", "打开相机"))
        self.actionSaveCaptions.setText(_translate("MainWindow", "保存结果"))
        self.action_Exit.setText(_translate("MainWindow", "&Exit"))
        self.actionInstruction.setText(_translate("MainWindow", "使用说明"))
        self.actionLoadVideo.setText(_translate("MainWindow", "输入视频"))
        self.actionFromVideo.setText(_translate("MainWindow", "输入视频"))

import images_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

