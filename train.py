import time
import os
import torch.autograd
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.metric_tool import get_mIoU
import numpy as np
working_path = os.path.abspath('.')
from collections import OrderedDict
from utils.loss import LatentSimilarity
from utils.utils import binary_accuracy as accuracy
from utils.utils import AverageMeter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

###################### Data and Model ########################
from models.SAM_Fusion2 import SAM_CD as Net
NET_NAME = '/path/of/model/name' 

from datasets import Levir_CD as RS
DATA_NAME = 'path/of/data/name'

########################## Parameters ########################
args = {
    'train_batch_size': 8,
    'val_batch_size':8,
    'lr': 0.001,
    'epochs': 200,
    'gpu': True,
    'dev_id': 0,
    'multi_gpu': None,  #"0,1,2,3",
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 50,
    'predict_step': 5,
    'img_size': 256,
    'crop_size': 512,
    'load_premodel':True,
    'pred_dir': os.path.join(working_path, 'results', NET_NAME, DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME),
    'chkpt_path': '/path/to/pretrain/model',  # pretrain model path
    'load_path': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, 'xxx.pth')}
########################## Parameters ########################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
writer = SummaryWriter(args['log_dir'])


def main():
    net = Net()
    #net.load_state_dict(torch.load(args['load_path']), strict=False)
    if args['multi_gpu']:
        net = torch.nn.DataParallel(net, [int(id) for id in args['multi_gpu'].split(',')])
    net.to(device=torch.device('cuda', int(args['dev_id'])))

    if args['load_premodel']:
        state_dict = torch.load(args["chkpt_path"], map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:] # remove `module.`
            if 'segmenter2' not in k:
                if 'module.' in k:
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict = state_dict
        net.load_state_dict(new_state_dict,strict=False)
        net.to(torch.device('cuda', int(args['dev_id']))).eval()


    train_set = RS.RS('train', random_crop=False, crop_nums=10, crop_size=args['crop_size'], random_flip=True) #'5_train_supervised',
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_set = RS.RS('test', sliding_crop=False, crop_size=args['crop_size'], random_flip=False)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), args['lr'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)

    train(train_loader, net, optimizer, val_loader)
    writer.close()
    print('Training finished.')

def train(train_loader, net, optimizer, val_loader):
    bestF = 0.0
    bestacc = 0.0
    bestIoU = 0.0
    bestloss = 1.0
    bestaccT = 0.0

    curr_epoch = 0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    criterion_sem = LatentSimilarity(T=3.0).to(torch.device('cuda', int(args['dev_id'])))
    T1 = 0
    T2 = 0
    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        train_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, args)
            imgs_A, imgs_B, labels = data
            if args['gpu']:
                imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
                imgs_B = imgs_B.to(torch.device('cuda', int(args['dev_id']))).float()
                labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs1, outputs2, outputs3, outA, outB = net(imgs_A, imgs_B)
            # outputs = net(imgs_A, imgs_B)
            assert outputs1.shape[1] == 1
            loss_bn1 = F.binary_cross_entropy_with_logits(outputs1, labels)
            loss_bn2 = F.binary_cross_entropy_with_logits(outputs2, labels)
            loss_bn3 = F.binary_cross_entropy_with_logits(outputs3, labels)

            loss_t = criterion_sem(outA, outB, labels)
            loss = loss_bn1 + loss_bn2 + loss_bn3 + loss_t
            # loss = loss_bn
            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            outputs = outputs3.cpu().detach()
            preds = F.sigmoid(outputs).numpy()
            acc_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, precision, recall, F1, IoU = accuracy(pred, label)
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_loss.val, acc_meter.val * 100))
                writer.add_scalar('train loss', train_loss.val, running_iter)
                loss_rec = train_loss.val
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        val_F, val_acc, val_IoU, val_loss, val_precision, val_recall = validate(val_loader, net, curr_epoch)

        # T1 = T1 + t1
        # T2 = T2 + t2
        if val_F > bestF:
            bestF = val_F
            bestacc = val_acc
            bestIoU = val_IoU
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME + '_e%d_P%.2f_R%.2f_F%.2f.pth' % (
            curr_epoch, val_precision * 100, val_recall * 100, val_F * 100)))
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        print('[epoch %d/%d %.1fs] Best rec: Train %.2f, Val %.2f, F1 score: %.2f IoU %.2f' \
              % (curr_epoch, args['epochs'], time.time() - begin_time, bestaccT * 100, bestacc * 100, bestF * 100,
                 bestIoU * 100))
        curr_epoch += 1
        # print('F1 from CD1: ' + str(T1) + ', F1 from CD2: ' + str(T2))
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()

    preds = []
    GTs = []

    T1 = 0
    T2 = 0
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels = data

        if args['gpu']:
            imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
            imgs_B = imgs_B.to(torch.device('cuda', int(args['dev_id']))).float()
            labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

        with torch.no_grad():
            output1, output2, output3, _, _ = net(imgs_A, imgs_B)
            loss_bn1 = F.binary_cross_entropy_with_logits(output1, labels)
            loss_bn2 = F.binary_cross_entropy_with_logits(output2, labels)
            loss_bn3 = F.binary_cross_entropy_with_logits(output3, labels)

            loss = loss_bn1 + loss_bn2 + loss_bn3

            output3 = F.sigmoid(output3)

        val_loss.update(loss.cpu().detach().numpy())

        outputs3 = output3.cpu().detach().numpy()

        labels = labels.cpu().detach().numpy()
        for (pred3, label) in zip(outputs3, labels):

            outputs3 = pred3.squeeze() > 0.5
            outputs3 = outputs3.astype(np.int64)
            preds.append(outputs3)
            label = label.astype(np.int64)
            GTs.append(label)


    score_dict = get_mIoU(2, GTs, preds)
    val_F, val_acc, val_IoU, val_precision, val_recall = score_dict['F1_1'], score_dict['acc'], score_dict['iou_1'], score_dict['precision_1'], score_dict['recall_1']
    curr_time = time.time() - start
    print('%.1fs Val loss %.2f Acc %.2f F %.2f P %.2f R %.2f' % (
    curr_time, val_loss.average(), val_acc * 100, val_F * 100, val_precision * 100, val_recall * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', val_acc, curr_epoch)

    # return val_F, val_acc, val_IoU, val_loss.avg, T1, T2
    return val_F, val_acc, val_IoU, val_loss.avg, val_precision, val_recall



def adjust_lr(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 3.0)
    running_lr = args['lr'] * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
