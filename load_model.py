import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

checkpoint = torch.load('penet_best.pth.tar', map_location=torch.device('cpu'))
torch.save(checkpoint, 'model_1')
