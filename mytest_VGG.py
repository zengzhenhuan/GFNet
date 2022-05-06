import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc
from Code.model_lung_infection.mynet_VGG import mynetVGG as Network
from Code.utils.dataloader_LungInf import test_dataset



def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='mynet_VGG',
                        help='change different backbone, choice: VGGNet16, ResNet50, Res2Net50')
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--data_path', type=str, default='./Dataset/TestingSet/covid/',
                        help='Path to test data')
    parser.add_argument('--pth_path', type=str, default='./Snapshots/save_weights/Semi-mynet_VGG/mynet_VGG-100.pth')
    parser.add_argument('--save_path', type=str, default='./Results/Lung infection segmentation/Semi-mynet_VGG/')
    opt = parser.parse_args()

    model = Network()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]) # uncomment it if you have multiply GPUs.
    model.load_state_dict(torch.load(opt.pth_path),strict=False)
    model.cuda()
    model.eval()

    image_root = '{}/Imgs/'.format(opt.data_path)
    gt_root = '{}/GT/'.format(opt.data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    os.makedirs(opt.save_path, exist_ok=True)

    MSE_minavg = 1
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, highmap, final_map ,lateral_edge = model(image)
        res = final_map
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.data.cpu().numpy().squeeze()
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(opt.save_path + name, res)

        loss_mse = torch.nn.MSELoss(reduction='mean')
        pred = torch.tensor(res, dtype=torch.float)
        target = torch.tensor(gt, dtype=torch.float)
        MSE = loss_mse(pred, target)
        if MSE < MSE_minavg:
            MSE_minavg = MSE
        print('MSE_minavg:', MSE_minavg)

    print('Test Done!')


if __name__ == "__main__":
    inference()