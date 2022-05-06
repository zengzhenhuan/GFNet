import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from Code.utils.dataloader_LungInf import get_loader
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, train_save):
    model.train()
    l1max = []
    l2max = []
    l3max = []
    l4max = []
    l5max = []
    l6max = []
    l7max = []
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]    # replace your desired scale, try larger scale for better accuracy in small object
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5, loss_record6,loss_record7,loss_record8 =  AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            edges = Variable(edges).cuda()
            # ---- rescaling the inputs (img/gt/edges) ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges= F.upsample(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, highmap, lateral_edge = model(images)  #
            # ---- loss function ----

            loss5 = joint_loss(lateral_map_5, gts)
            loss4 = joint_loss(lateral_map_4, gts)
            loss3 = joint_loss(lateral_map_3, gts)
            loss2 = joint_loss(lateral_map_2, gts)
            loss1 = joint_loss(lateral_map_1, gts)
            loss6 = joint_loss(highmap, gts)
            loss7 = joint_loss(lateral_edge, edges)
            loss = loss1 + loss2  + loss3 + loss4 + loss5 + loss6 +loss7

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
                loss_record6.update(loss6.data, opt.batchsize)
                loss_record7.update(loss7.data, opt.batchsize)
        # ---- train logging ----
        if i % 1 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral_map_1: {:.4f}, '
                  'lateral_map_2: {:.4f},lateral_map_3: {:0.4f}, lateral_map_4: {:0.4f}, lateral_map_5: {:0.4f}, highmap: {:0.4f},lateral_edge: {:0.4f}],'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(),
                         loss_record2.show(),loss_record3.show(), loss_record4.show(), loss_record5.show(), loss_record6.show(),loss_record7.show()))
            l1max.append(loss1.detach().cpu().numpy())
            l2max.append(loss2.detach().cpu().numpy())
            l3max.append(loss3.detach().cpu().numpy())
            l4max.append(loss4.detach().cpu().numpy())
            l5max.append(loss5.detach().cpu().numpy())
            l6max.append(loss6.detach().cpu().numpy())
            l7max.append(loss7.detach().cpu().numpy())
        if i == total_step:
            l1.append(max(l1max))     #S1
            print(l1max)
            print(l1)
            l2.append(max(l2max))     #S2
            l3.append(max(l3max))     #S3
            l4.append(max(l4max))     #S4
            l5.append(max(l5max))     #S5
            l6.append(max(l6max))     #Sg
            l7.append(max(l7max))     #Eg

    # ---- save model_lung_infection ----
    save_path = './Snapshots/save_weights/{}/'.format(train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'mynet_VGG-%d.pth' % (epoch+1))
        print('[Saving Snapshot:]', save_path + 'mynet_VGG-%d.pth' % (epoch+1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--epoch', type=int, default=100 ,
                        help='epoch number')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate')
    parser.add_argument('--batchsize', type=int, default=5,
                        help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='set the size of training sample')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='every n epochs decay learning rate')
    parser.add_argument('--is_thop', type=bool, default=True,
                        help='whether calculate FLOPs/Params (Thop)')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='choose which GPU device you want to use')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers in dataloader. In windows, set num_workers=0')
    # model_lung_infection parameters
    parser.add_argument('--net_channel', type=int, default=32,
                        help='internal channel numbers in the Inf-Net, default=32, try larger for better accuracy')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='binary segmentation when n_classes=1')
    parser.add_argument('--backbone', type=str, default='mynet_VGG',
                          help='change different backbone, choice: VGGNet16, ResNet50, Res2Net50,mynet')
    # training dataset
    parser.add_argument('--train_path', type=str,
                        default='./Dataset/TrainingSet/LungInfection-Train/Doctor-label')
    parser.add_argument('--is_semi', type=bool, default=False,
                        help='if True, you will turn on the mode of `Semi-Inf-Net`')
    parser.add_argument('--is_pseudo', type=bool, default=False,
                        help='if True, you will train the model on pseudo-label')
    parser.add_argument('--train_save', type=str, default=None,
                        help='If you use custom save path, please edit `--is_semi=False` and `--is_pseudo=False`')

    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(opt.gpu_device)

    if  opt.backbone == 'mynet_VGG':
        print('Backbone loading: mynet_VGG')
        from Code.model_lung_infection.mynet_VGG import mynetVGG
    else:
        raise ValueError('Invalid backbone parameters: {}'.format(opt.backbone))
    model = mynetVGG(channel=opt.net_channel, n_class=opt.n_classes).cuda()

    if opt.is_semi and opt.backbone == 'mynet_VGG':
        print('Loading weights from weights file trained on pseudo label')
        model.load_state_dict(torch.load('./Snapshots/save_weights/mynet_VGG_Pseudo/mynet_VGG-100.pth'),strict=False)
    else:
        print('Not loading weights from weights file')

    # weights file save path
    if opt.is_pseudo and (not opt.is_semi):
        train_save = 'mynet_VGG_Pseudo'
    elif (not opt.is_pseudo) and opt.is_semi:
        train_save = 'Semi-mynet_VGG'
    elif (not opt.is_pseudo) and (not opt.is_semi):
        train_save = 'mynet_VGG'
    else:
        print('Use custom save path')
        train_save = opt.train_save

    # ---- calculate FLOPs and Params ----
    if opt.is_thop:
        from Code.utils.utils import CalParams
        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()
        CalParams(model, x)

    # ---- load training sub-modules ----
    BCE = torch.nn.BCEWithLogitsLoss()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root,edge_root,
                              batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=opt.num_workers)
    total_step = len(train_loader)

    # ---- start !! -----
    epoch1=[]
    l1=[]
    l2=[]
    l3=[]
    l4=[]
    l5=[]
    l6=[]
    l7=[]

    for epoch in tqdm(range(1,opt.epoch)):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, train_save)
        epoch1.append(epoch)
    plt.plot(epoch1, l1,label="S1")
    plt.plot(epoch1, l2,label="S2")
    plt.plot(epoch1, l3, label="S3")
    plt.plot(epoch1, l4, label="S4")
    plt.plot(epoch1, l5, label="S5")
    plt.plot(epoch1, l6, label="Sg")
    plt.plot(epoch1, l7,label="Eg")
    plt.axis([0, 100, 0, 2])
    plt.title("Loss after epoch 100")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
