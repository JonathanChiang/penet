"""Take in DICOM file and output CTPA to into Model
"""

import argparse
import util
import torch
from saver import ModelSaver
from models import *

input_study = '/data/location/'
series_description="CTA 2.0 CTA/PULM CE"
ckpt_path = 'penet_best.pth.tar'
device = 'cuda'
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
model, ckpt_info = ModelSaver.load_model(ckpt_path, '1')

print ("Sending model to GPU device...")
#start_epoch = ckpt_info['epoch'] + 1
model = model.to(device)

print ("Evaluating study...")
model.eval()

predicted_probabilities = []    # window wise for a single study
with torch.no_grad():
    for window in study_windows: 
        cls_logits = model.forward(window.to(device, dtype=torch.float))
        cls_probs = torch.sigmoid(cls_logits).to('cpu').numpy()
        predicted_probabilities.append(cls_probs[0][0])

print (f"Probablity of having Pulmonary Embolism: {max(predicted_probabilities)}")



'''
if __name__ == "__main__": 

    # parse in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_study", type=str, default="/data4/intermountain/CTPA/CTPA_RANDOM_DICOM/1770659")
    parser.add_argument("--series_description", type=str, default="CTA 2.0 CTA/PULM CE")
    parser.add_argument("--ckpt_path", type=str, default="/data4/PE_stanford/ckpts/best.pth.tar")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args = parser.parse_args()

    if ","  in args.gpu_ids:
        args.gpu_ids = args.gpu_ids.split(",")
    else:
        args.gpu_ids = [int(args.gpu_ids)]

    main(args) 
'''
