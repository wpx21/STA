import argparse
from sta import STA
from PIL import Image, ImageFont, ImageDraw
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
import pretrainedmodels
from utils import *


parser = argparse.ArgumentParser("Transfer Attack")
parser.add_argument('--method', type=str, default='naa', help='transfer_based attack methods')
parser.add_argument('--model', type=str, default='inceptionv4', help='attack model')
parser.add_argument('--targeted', action='store_true', default=False, help='targeted attack')
parser.add_argument('--steps', type=int, default=5, help='steps')
parser.add_argument('--alp_scale', type=float, default=120, help='alp_scale')
parser.add_argument('--eps', type=int, default=10, help='eps')
parser.add_argument('--num_block', type=int, default=1, help='num_block')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

step_size = args.eps / (255 * args.steps), alpha = args.eps / args.steps

Attacker = STA(model_name=args.model, epsilon=args.eps / 255., alpha=alpha /255., alp_scale=120.,num_block=args.num_block, epoch=args.steps, targeted=args.targeted, device=device)

print("Attacking {} with {} under {}/255 norm !".format(args.model, args.method, args.eps))

DATASET_PATH = 'Your Dataset Path'
ADV_PATH = 'Saved Adv Path'

target_model = "Your Model"
target_model.eval().to(device)

def attack_process(img_path):
    ori_img = Variable(image_loader(img_path), requires_grad=True).to(device)
    ori_lbl = target_model(ori_img).argmax(dim=1)
    per_img = Attacker(ori_img, ori_lbl)
    adv_img = ori_img + per_img
    return adv_img

if __name__ == '__main__':
    image = 'Your Image'
    adv_img = attack_process(image)
    ### Your code here ###

