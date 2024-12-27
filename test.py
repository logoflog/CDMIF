# -*- coding:utf-8 -*-
# @Author: Luo Ge
# @Email: geyouguang@163.com
# @File : test.py
# @Time : 2024/11/21

import os
import torch
from dataloader import loader
from torch.utils.data import DataLoader
from CDMIF import CDMIF
from metrics import *

# *-----------------*
# hyper-parameters
srcadir = r'F:\dataset\multi-exposure\sample\SourceA'
srcbdir = r'F:\dataset\multi-exposure\sample\SourceB'
exist_gt = True
gtdir = r'F:\dataset\multi-exposure\sample\Label'
# /root/autodl-tmp/dataset/multi-exposure/testset1/Label

imgshape = (500, 500)  # (W, H)
gtdir = srcadir if exist_gt == False else gtdir
fusedir = r'./Fused/'
trainedpth = 'model/4.pth'
# *-----------------*

class Tester():
    def __init__(self):
        print("=====> Loading testset")
        test_set = loader(srcadir, srcbdir, gtdir)
        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

        if (os.path.exists(trainedpth)):
            state = torch.load(trainedpth)
            self.model = CDMIF().cuda()
            self.model.load_state_dict(state['model'])
            self.model.eval()

    def test(self):
        with torch.no_grad():
            for batch, (srcapath, srcbpath, gtpath) in enumerate(self.test_loader):
                # print(srcapath,srcbpath,gtpath)
                X1, X2 = [], []
                shapelist = {}
                # 这个循环不去除，用来添加维度
                for i in range(len(srcapath)):
                    # print(srcapath[i])
                    img_X1 = cv2.imread(srcapath[i])
                    h, w, _ = img_X1.shape
                    shapelist[srcapath[i]] = (w, h)
                    X1.append(np.expand_dims(cv2.resize(img_X1, imgshape, interpolation=cv2.INTER_LINEAR), 0))
                    img_X2 = cv2.imread(srcbpath[i])
                    X2.append(np.expand_dims(cv2.resize(img_X2, imgshape, interpolation=cv2.INTER_LINEAR), 0))
                X1 = torch.from_numpy(np.concatenate(X1, axis=0)).float().cuda()
                X2 = torch.from_numpy(np.concatenate(X2, axis=0)).float().cuda()

                F = self.model(X1, X2).cpu().numpy()
                for i in range(len(srcapath)):
                    img = cv2.resize(F[i], shapelist[srcapath[i]])
                    os.makedirs(fusedir, exist_ok=True)
                    fusedfilename = srcapath[i].split('/')[-1].split('.')[0]
                    cv2.imwrite(fusedir + fusedfilename + '.png', img)

                print('Current/Total:{}/{}'.format(batch + 1, len(self.test_loader)))

    def calmetric_gt(self):
        # metrics
        en_list = []
        psnr_list = []
        ag_list = []
        ssim_list = []
        cc_list = []
        vif_list = []
        sd_list = []
        nabf_list = []

        for batch, (srcapath, srcbpath, gtpath) in enumerate(self.test_loader):
            X1 = cv2.imread(srcapath[0])
            X2 = cv2.imread(srcbpath[0])
            Gt = cv2.imread(gtpath[0])
            fusedfilename = srcapath[0].split('/')[-1].split('.')[0]
            F = cv2.imread(fusedir + fusedfilename + '.png')

            psnr_val = calculate_psnr(F, Gt)
            psnr_list.append(psnr_val)
            ag_val = average_gradient(F)
            ag_list.append(ag_val)
            ssim_val = calculate_ssim(F, Gt)
            ssim_list.append(ssim_val)
            cc_val = correlation_coefficient(F, Gt)
            cc_list.append(cc_val)
            vif_val = calculate_vif(F, Gt)
            vif_list.append(vif_val)
            en_val = entropy(F)
            en_list.append(en_val)
            sd_val = standard_deviation(F)
            sd_list.append(sd_val)
            nabf_val = nabf(X1, X2, F)
            nabf_list.append(nabf_val)

            print(
                'Current/Total:{}/{} \t En:{:.4f} \t SD:{:.4f} \t Nabf:{:.4f} \t PSNR:{:.4f} \t AG:{:.4f} \t SSIM:{:.4f} \t CC:{:.4f} \t VIF:{:.4f}'
                .format(batch + 1, len(self.test_loader), en_val, sd_val, nabf_val, psnr_val, ag_val, ssim_val, cc_val,
                        vif_val))

        print('Mean value of En', np.mean(en_list))
        print('Mean value of SD', np.mean(sd_list))
        print('Mean value of Nabf', np.mean(nabf_list))
        print('Mean value of PSNR', np.mean(psnr_list))
        print('Mean value of AG', np.mean(ag_list))
        print('Mean value of SSIM', np.mean(ssim_list))
        print('Mean value of CC', np.mean(cc_list))
        print('Mean value of VIF', np.mean(vif_list))

    def calmetric(self):
        # metrics
        en_list = []
        mpsnr_list = []
        ag_list = []
        mssim_list = []
        mcc_list = []
        mvif_list = []
        sd_list = []
        nabf_list = []

        for batch, (srcapath, srcbpath, gtpath) in enumerate(self.test_loader):
            X1 = cv2.imread(srcapath[0])
            X2 = cv2.imread(srcbpath[0])
            name = srcapath[0].split('/')[-1].split('.')[0] + '.png'
            F = cv2.imread(fusedir + name)

            psnr_val = (calculate_psnr(X1, F) + calculate_psnr(X2, F)) / 2
            mpsnr_list.append(psnr_val)
            ag_val = average_gradient(F)
            ag_list.append(ag_val)
            ssim_val = (calculate_ssim(X1, F) + calculate_ssim(X2, F)) / 2
            mssim_list.append(ssim_val)
            cc_val = (correlation_coefficient(X1, F) + correlation_coefficient(X2, F)) / 2
            mcc_list.append(cc_val)
            vif_val = (calculate_vif(X1, F) + calculate_vif(X2, F)) / 2
            mvif_list.append(vif_val)
            en_val = entropy(F)
            en_list.append(en_val)
            sd_val = standard_deviation(F)
            sd_list.append(sd_val)
            nabf_val = nabf(X1, X2, F)
            nabf_list.append(nabf_val)
            
            print(
                'Current/Total:{}/{} \t En:{:.4f} \t SD:{:.4f} \t Nabf:{:.4f} \t mPSNR:{:.4f} \t AG:{:.4f} \t mSSIM:{:.4f} \t mCC:{:.4f} \t mVIF:{:.4f}'
                .format(batch + 1, len(self.test_loader), en_val, sd_val, nabf_val, psnr_val, ag_val, ssim_val, cc_val,
                        vif_val))

        print('Mean value of En', np.mean(en_list))
        print('Mean value of SD', np.mean(sd_list))
        print('Mean value of Nabf', np.mean(nabf_list))
        print('Mean value of mPSNR', np.mean(mpsnr_list))
        print('Mean value of AG', np.mean(ag_list))
        print('Mean value of mSSIM', np.mean(mssim_list))
        print('Mean value of mCC', np.mean(mcc_list))
        print('Mean value of mVIF', np.mean(mvif_list))


if __name__ == '__main__':
    tester = Tester()
    tester.test()
    # if exist_gt:
    #     tester.calmetric_gt()
    # else:
    #     tester.calmetric()