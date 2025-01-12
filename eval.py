
import os
from utils.metric_tool import get_mIoU
import cv2
import numpy as np
import torch.autograd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
working_path = os.path.abspath('.')

from collections import OrderedDict
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

###################### Data and Model ########################
from models.SAM_Fusion2 import SAM_CD as Net
NET_NAME = '/path/of/model/name' 

from datasets import Levir_CD as RS
DATA_NAME = '/path/of/data/name'

########################## Parameters ########################
args = {
    'train_batch_size': 4,
    'val_batch_size':1,
    'lr': 0.1,
    'epochs': 50,
    'gpu': True,
    'dev_id': 0,
    'multi_gpu': None,  #"0,1,2,3",
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 50,
    'predict_step': 5,
    'crop_size': 256,
    'chkpt_path':os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, '/path/to/ckpt'),
    'pred_dir': os.path.join(working_path, 'results', NET_NAME, DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', NET_NAME, DATA_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, 'xxx.pth')}
########################## Parameters ########################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
writer = SummaryWriter(args['log_dir'])
def main():
    net = Net()
    #net.load_state_dict(torch.load(args['load_path']), strict=False)
    net.to(device=torch.device('cuda', int(args['dev_id'])))
    val_set = RS.RS('test', sliding_crop=False, crop_size=args['crop_size'], random_flip=False)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
    state_dict = torch.load(args["chkpt_path"], map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict
    net.load_state_dict(new_state_dict)
    net.to(torch.device('cuda', int(args['dev_id']))).eval()
    validate(val_loader,net)


def validate(val_loader, net):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()

    preds = []
    GTs = []
    for data in tqdm(val_loader):
    # for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels = data
        if args['gpu']:
            imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
            imgs_B = imgs_B.to(torch.device('cuda', int(args['dev_id']))).float()
            labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

        with torch.no_grad():
            output1, output2, output3, _,_ = net(imgs_A, imgs_B)
            output3 = F.sigmoid(output3)

        outputs3 = output3.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        for (pred3, label) in zip(outputs3, labels):

            preds.append(pred3)
            label = label.astype(np.int64)
            GTs.append(label)
    score_dict = get_mIoU(2, GTs, preds)
    F1, acc, IoU, precision, recall = score_dict['F1_1'], score_dict['acc'], score_dict['iou_1'], score_dict['precision_1'], score_dict['recall_1']
    print('ACC: ' + str(acc))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(F1))
    print('IoU: ' + str(IoU))


if __name__ == '__main__':
    main()
