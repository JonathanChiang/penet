import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import util
from saver import ModelSaver
from models import *

input_study = '/data/location/'
series_description="CTA 2.0 CTA/PULM CE"
ckpt_path = 'penet_best.pth.tar'
device = 'cpu'
map_location='cpu'
print("Reading input dicom...")
study = util.dicom_2_npy(input_study)
print('is study empty')
print(study)
# normalize and convert to tensor
print("Formatting input for model...")
study_windows = util.format_img(study) 
print("is study window empty")
print(study_windows)

print ("Loading saved model...")
model, ckpt_info = ModelSaver.load_model(ckpt_path, 0)

print ("Sending model to GPU device...")
#start_epoch = ckpt_info['epoch'] + 1
model = model.to(device)



dummy_input = torch.randn(64, 3, 7, 7, 7)
torch.onnx.export(model.module, dummy_input, "penet.onnx", verbose=True)
