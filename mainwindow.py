# -*- coding: utf-8 -*-
import time
import sys
import os
import math
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from Ui_mainwindow import Ui_MainWindow
from NetworkThread import LoadCNNThread, LoadLSTMThread, LoadImagesThread, CaptionThread
import argparse
from six.moves import cPickle
class MainWindow(QMainWindow,  Ui_MainWindow):
    #loadmodelSignal = pyqtSignal(str,  str)
    imagepathSignal = pyqtSignal(list)
    changeBatchSignal = pyqtSignal()
    def __init__(self,  parent=None):
        super(MainWindow,  self).__init__(parent)
        self.setupUi(self)
        self.num_images = 0
        self.images = []
        self.ids = []
        self.current_batch = []
        self.cnn = None
        self.lstm = None
        self.vocab = None
        self.opt = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.auto_check)

        #menu action
        self.actionLoadModel.triggered.connect(self.loadmodel)
        self.actionFromFile.triggered.connect(self.loadimage)
        self.actionFromCamera.triggered.connect(self.loadcamera)
        self.actionSaveCaptions.triggered.connect(self.savecaptions)
        self.action_Exit.triggered.connect(self.close)
        self.actionInstruction.triggered.connect(self.showhelp)
        
        #button action
        self.batch_id = 1
        self.changeBatchSignal.connect(self.show_images)
        self.pushButton_last.clicked.connect(self.getLastBatch)
        self.pushButton_next.clicked.connect(self.getNextBatch)
        self.pushButton_generate.clicked.connect(self.generateCaptions)
        self.pushButton_autostart.clicked.connect(self.autostart)
        self.pushButton_autostop.clicked.connect(self.autostop)


        #增加切换不同模型时线程的对应处理     
        #cnn和lstm采用单独线程读入
        self.cnn_model_path = "no_cnn_get"
        self.lstm_model_path = "no_lstm_get"
        self.LoadCNNThread = LoadCNNThread()
        self.LoadCNNThread.cnnSignal.connect(self.getcnn)
        self.LoadLSTMThread = LoadLSTMThread()
        self.LoadLSTMThread.lstmSignal.connect(self.getlstm)

        #load image
        self.LoadImagesThread = LoadImagesThread()
        self.LoadImagesThread.getimageSignal.connect(self.getimages)
        
        #caption thread
        self.CaptionThread = CaptionThread()
        self.CaptionThread.resultSignal.connect(self.updatecaptions)
        #self.CaptionThread.finished.connect(self.auto_check) 
    def setup_opt(self, infos_path='./log_adaatt/infos_adaatt.pkl'):
        #模型默认参数
        self.opt = argparse.Namespace()
        self.opt.infos_path = infos_path
        self.opt.batch_size = 8
        self.opt.num_images = -1
        self.opt.sample_max = 1
        self.opt.beam_size = 2
        self.opt.temperature = 1.0
        self.opt.input_fc_dir = ''
        self.opt.input_att_dir = ''
        self.opt.input_label_h5 = ''
        self.opt.image_folder = ''
        self.opt.input_json = ''
        self.opt.id = ''
         
        self.infos = cPickle.load(open(self.opt.infos_path, 'r'))
        if len(self.opt.input_fc_dir)==0:
            self.opt.input_fc_dir = self.infos['opt'].input_fc_dir
            self.opt.input_att_dir = self.infos['opt'].input_att_dir
            self.opt.input_label_h5 = self.infos['opt'].input_label_h5
        if len(self.opt.input_json)==0:
            self.opt.input_json = self.infos['opt'].input_json
        if len(self.opt.id)==0:
            self.opt.id = self.infos['opt'].id                                          
        ignore = ['id',  'batch_size',  'beam_size',  'start_from', 'language_eval']
        for k in vars(self.infos['opt']).keys():
            if k not in ignore:
                if k in vars(self.opt):
                    assert vars(self.opt)[k] == vars(self.infos['opt'])[k],  k+' option not consistent'
                else:
                    vars(self.opt).update({k:vars(self.infos['opt'])[k]})
        self.vocab = self.infos['vocab']

    def loadmodel(self):
        #load network
        self.cnn_model_path, _ = QFileDialog.getOpenFileName(self,  "加载CNN网络模型",  "./ui_models",  "Pytorch Model (*.pth *.pth.tar)")
        if self.cnn_model_path:
            if "resnet101" in self.cnn_model_path:
                self.LoadCNNThread.loadmodelfromdir(self.cnn_model_path, "resnet101", 1000)
            elif "resnet152" in self.cnn_model_path:
                self.LoadCNNThread.loadmodelfromdir(self.cnn_model_path, "resnet152", 62)
        infos_path, _ = QFileDialog.getOpenFileName(self, "加载网络结构信息", "./ui_models", "cPickle File (*.pkl)")
        if infos_path:
            self.setup_opt(infos_path)

        self.lstm_model_path,  _ = QFileDialog.getOpenFileName(self,  "加载LSTM网络模型",  "./ui_models",  "Pytorch Model (*.pth *pth.tar)")
        if self.lstm_model_path:
            self.LoadLSTMThread.loadmodelfromdir(self.lstm_model_path, self.opt)

    def getcnn(self, cnn):
        self.cnn = cnn
        self.statusBar.showMessage("CNN model loading complete!")

    def getlstm(self, lstm):
        self.lstm = lstm
        self.statusBar.showMessage("LSTM model loading complete!")
    
    def loadimage(self):
        #load local image
        self.image_path = ''
        self.image_path = QFileDialog.getExistingDirectory(self,  "打开图片所在路径",  "./")
        if self.image_path:
            self.LoadImagesThread.loadimagesfromdir(self.image_path)
    
    def getimages(self, images, ids):
        self.images = images
        self.ids = ids
        self.batch_id = 1
        self.num_images = len(self.images)
        self.changeBatchSignal.emit()
        
    def loadcamera(self):
        #load from camera
        pass
        
    def savecaptions(self):
        #save all caption results
        pass
        
    def showhelp(self):
        #show how to use this software
        QMessageBox.information(self,  "Help Message",  "1、加载网络：\t文件->网络模型->选择文件\n2、输入图片：\t文件->图片来源->打开路径（or 打开相机）",  QMessageBox.Ok,  QMessageBox.Ok)
    
    def getLastBatch(self):
        if self.batch_id <= 1:
            return
        else:
            self.batch_id = self.batch_id - 1
            self.changeBatchSignal.emit()
    
    def getNextBatch(self):
        if self.batch_id >= math.ceil(self.num_images / 8):
            return
        else:
            self.batch_id = self.batch_id + 1
            self.changeBatchSignal.emit()

    def autostart(self):
        self.autoFlag = True
        self.autocaption()

    def autostop(self):
        self.autoFlag = False
        self.pushButton_last.setDisabled(False)
        self.pushButton_next.setDisabled(False)
        self.pushButton_generate.setDisabled(False)
        self.pushButton_autostart.setDisabled(False)
    def auto_check(self):
        if self.autoFlag:
            self.getNextBatch()
            self.autocaption()
        else:
            return
    def autocaption(self):
        if self.cnn and self.lstm and self.current_batch:
            if self.autoFlag:
                self.pushButton_last.setDisabled(True)
                self.pushButton_next.setDisabled(True)
                self.pushButton_generate.setDisabled(True)
                self.pushButton_autostart.setDisabled(True)
                self.timer.start(4000)
                self.generateCaptions()
        else:
            return

    def generateCaptions(self):
        if self.cnn and self.lstm and self.current_batch:
            self.CaptionThread.captionfromimgbatch(self.cnn, self.lstm, self.current_batch, self.vocab, self.opt)
        else:
            pass
    def updatecaptions(self,  sents=[]):
        while len(sents) < 8:
            sents.append('')
        self.label_1.setText('1. '+sents[0])
        self.label_2.setText('2. '+sents[1])
        self.label_3.setText('3. '+sents[2])
        self.label_4.setText('4. '+sents[3])
        self.label_5.setText('5. '+sents[4])
        self.label_6.setText('6. '+sents[5])
        self.label_7.setText('7. '+sents[6])
        self.label_8.setText('8. '+sents[7])
    
    def show_images(self):
        #show the current batch of images on mainwindow
        self.current_batch = []
        current_images = []
        print("Batch ID: " + str(self.batch_id) + '/'+str(math.ceil(self.num_images/8)))
        if self.batch_id >= 1 and self.batch_id <= math.ceil(self.num_images/8) and self.images:
            if self.num_images >= self.batch_id * 8:
                self.current_batch = self.images[(self.batch_id-1)*8 : (self.batch_id*8)]
            else:
                # remaining images is not enough for a new batch, padding with black image
                remaining_num = self.num_images - (self.batch_id-1) * 8
                self.current_batch = self.images[(self.batch_id-1)*8 : ]
                padding_images = [":/img/newblack.jpg" for i in range(8-remaining_num)]
                self.current_batch.extend(padding_images)
            for imgname in self.current_batch:
                img = QImage(imgname)
                img = img.scaled(200,  200,  Qt.IgnoreAspectRatio,  Qt.SmoothTransformation)
                current_images.append(img)
            self.image_1.setPixmap(QPixmap(current_images[0]))
            self.image_2.setPixmap(QPixmap(current_images[1]))
            self.image_3.setPixmap(QPixmap(current_images[2]))
            self.image_4.setPixmap(QPixmap(current_images[3]))
            self.image_5.setPixmap(QPixmap(current_images[4]))
            self.image_6.setPixmap(QPixmap(current_images[5]))
            self.image_7.setPixmap(QPixmap(current_images[6]))
            self.image_8.setPixmap(QPixmap(current_images[7]))
        
        #clean the caption area
        self.updatecaptions()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
