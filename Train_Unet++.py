import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from Code.utils.dataloader_LungInf import get_loader
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from tqdm import tqdm

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
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]    # replace your desired scale, try larger scale for better accuracy in small object
    loss_record =  AvgMeter()
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
            final = model(images)  #
            # ---- loss function ----


            loss = joint_loss(final, gts)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        # ---- train logging ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], final: {:.4f}, '.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(),
                         ))

    # ---- save model_lung_infection ----
    save_path = './Snapshots/save_weights/{}/'.format(train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'UNet++-%d.pth' % (epoch+1))
        print('[Saving Snapshot:]', save_path + 'UNet++-%d.pth' % (epoch+1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--epoch', type=int, default=100 ,
                        help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--batchsize', type=int, default=2,
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
    parser.add_argument('--net_channel', type=int, default=3,
                        help='internal channel numbers in the Inf-Net, default=32, try larger for better accuracy')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='binary segmentation when n_classes=1')
    parser.add_argument('--backbone', type=str, default='UNet++',
                          help='change different backbone, choice: VGGNet16, ResNet50, Res2Net50,mynet')
    # training dataset
    parser.add_argument('--train_path', type=str,
                        default='./Dataset/TrainingSet/LungInfection-Train/covid')
    parser.add_argument('--is_semi', type=bool, default=True,
                        help='if True, you will turn on the mode of `Semi-Inf-Net`')
    parser.add_argument('--is_pseudo', type=bool, default=False,
                        help='if True, you will train the model on pseudo-label')
    parser.add_argument('--train_save', type=str, default=None,
                        help='If you use custom save path, please edit `--is_semi=False` and `--is_pseudo=False`')

    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(opt.gpu_device)

    if opt.backbone == 'Res2Net50':
        print('Backbone loading: Res2Net50')
        from Code.model_lung_infection.InfNet_Res2Net import Inf_Net
    elif opt.backbone == 'ResNet50':
        print('Backbone loading: ResNet50')
        from Code.model_lung_infection.InfNet_ResNet import Inf_Net
    elif opt.backbone == 'VGGNet16':
        print('Backbone loading: VGGNet16')
        from Code.model_lung_infection.InfNet_VGGNet import Inf_Net
    elif opt.backbone == 'UNet++':
        print('Backbone loading: UNet++')
        from Code.model_lung_infection.UNet2plus.UNet_2Plus import UNet_2Plus
    else:
        raise ValueError('Invalid backbone parameters: {}'.format(opt.backbone))
    model = UNet_2Plus(in_channels=opt.net_channel, n_classes=opt.n_classes, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True).cuda()


    if opt.is_semi and opt.backbone == 'UNet++':
        print('Loading weights from weights file trained on pseudo label')
        model.load_state_dict(torch.load('./Snapshots/save_weights/UNet++_Pseudo/UNet++-100.pth'),strict=False)
    else:
        print('Not loading weights from weights file')

    # weights file save path
    if opt.is_pseudo and (not opt.is_semi):
        train_save = 'UNet++_Pseduo'
    elif (not opt.is_pseudo) and opt.is_semi:
        train_save = 'Semi-UNet++'
    elif (not opt.is_pseudo) and (not opt.is_semi):
        train_save = 'UNet++'
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

    for epoch in tqdm(range(1,opt.epoch)):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, train_save)
