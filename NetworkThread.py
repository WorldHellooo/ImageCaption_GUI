from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import argparse
import skimage
import skimage.io
from PIL import Image
import os
import time
from six.moves import cPickle
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms as trn
preprocess = trn.Compose([
    trn.Resize(size=(256,256)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229,  0.224,  0.225])])

from misc.resnet_utils import myResnet
import misc.utils as utils
import misc.resnet
import models
from models import CaptionModel

class LoadCNNThread(QThread):
    cnnSignal = pyqtSignal(myResnet)
    def __init__(self):
        super(LoadCNNThread,  self).__init__()

    def loadmodelfromdir(self, cnn_model_path, cnn_model="resnet152", num_classes=62):
        self.cnn_model = cnn_model
        self.cnn_model_path = cnn_model_path
        self.num_classes = num_classes
        self.start()

    def run(self):
        if self.cnn_model_path != 'no_cnn_get':
            self.my_resnet = getattr(misc.resnet,  self.cnn_model)(num_classes=self.num_classes)
            cnn_model = torch.load(self.cnn_model_path)
            #self.my_resnet.load_state_dict(torch.load(self.cnn_model_path).state_dict())
            self.my_resnet.load_state_dict(torch.load(self.cnn_model_path))
            self.my_resnet = myResnet(self.my_resnet)
            self.my_resnet.cuda()
            self.my_resnet.eval()
            self.cnnSignal.emit(self.my_resnet)

class LoadLSTMThread(QThread):
    lstmSignal = pyqtSignal(CaptionModel)
    def __init__(self):
        super(LoadLSTMThread,  self).__init__()
    
    def loadmodelfromdir(self, lstm_model_path, opt):
        self.lstm_model_path = lstm_model_path
        self.opt = opt
        self.start()
    
    def run(self):
        if self.lstm_model_path != 'no_lstm_get':
            self.lstm_model = models.setup(self.opt)
            self.lstm_model.load_state_dict(torch.load(self.lstm_model_path))
            self.lstm_model.cuda()
            self.lstm_model.eval()
            self.lstmSignal.emit(self.lstm_model)

class LoadImagesThread(QThread):
    getimageSignal = pyqtSignal(list, list)
    def __init__(self):
	super(LoadImagesThread, self).__init__()
        
    def loadimagesfromdir(self, image_path):
        self.image_path = image_path
        self.start()

    def run(self):
        self.images = []
        self.ids = []
        def isImage(f):
            supportedExt = ['.jpg',  '.JPG',  '.jpeg',  '.JPEG',  '.png',  '.PNG',  '.ppm',  '.PPM']
            for ext in supportedExt:
                start_idx = f.rfind(ext)
                if start_idx >= 0 and start_idx + len(ext) == len(f):
                    return True
            return False
        n = 1
        for root, dirs,  files in os.walk(self.image_path,  topdown=False):
            for file in files:
                fullpath = os.path.join(self.image_path,  file)
                if isImage(fullpath):
                    self.images.append(fullpath)
                    self.ids.append(str(n))
                    n = n + 1
        self.getimageSignal.emit(self.images, self.ids)

class CaptionThread(QThread):
    resultSignal = pyqtSignal(list)
    
    def __init__(self):
        super(CaptionThread,  self).__init__()
        self.img_batch = []
        
    def run(self):
        #init network
        t_start = time.time()
        self.batch_size = len(self.img_batch)
        fc_batch = np.ndarray((self.batch_size,   2048),  dtype='float32')
        att_batch = np.ndarray((self.batch_size,  14,  14,  2048),  dtype='float32')
        infos = []
        t_cnn_start = time.time()
        minibatch = torch.FloatTensor(self.batch_size, 3, 256,256).cuda()
        for i in range(self.batch_size):
            img = Image.open(self.img_batch[i])
            #img = img.astype('float32')/255.0
            #minibatch = torch.from_numpy(img.transpose([2, 0, 1])).cuda()
            minibatch[i,:,:,:] = preprocess(img)
        
            info_struct = {}
            info_struct['id'] = str(i)
            info_struct['file_path'] = self.img_batch[i]
            infos.append(info_struct)
        with torch.no_grad():
            input = Variable(minibatch)
                #print img.shape
            tmp_fc,  tmp_att = self.my_resnet(input)
                #print tmp_fc.shape, tmp_att.shape
        fc_batch = tmp_fc.data.cpu().float().numpy()
        att_batch = tmp_att.data.cpu().float().numpy()
        data = {}
        data['fc_feats'] = fc_batch
        data['att_feats'] = att_batch
        data['infos']  = infos
        #t_cnn_end = time.time()

        with torch.no_grad():
            fc_feats = Variable(torch.from_numpy(fc_batch)).cuda()
            att_feats = Variable(torch.from_numpy(att_batch)).cuda()
            t_cnn_end = time.time()
            seq,  _ = self.lstm_model.sample(fc_feats,  att_feats,  vars(self.opt))
            sents = utils.decode_sequence(self.vocab,  seq)
            self.resultSignal.emit(sents)
            t_end = time.time()
            print "Caption current image batch cost: "+str(t_end-t_start)+"s"
            print "CNN process cost: "+str(t_cnn_end-t_cnn_start)+'s'
            print "LSTM process cost: "+str(t_end-t_cnn_end)+'s'
    def captionfromimgbatch(self,  cnn, lstm, img_batch, vocab, opt):
        self.my_resnet = cnn
        self.lstm_model = lstm
        self.img_batch = img_batch
        self.vocab = vocab
        self.opt = opt
        self.start()
        
