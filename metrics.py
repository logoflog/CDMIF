# -*- coding:utf-8 -*-
# @Author: Luo Ge
# @Email: geyouguang@163.com
# @File : metrics.py
# @Time : 2024/11/18

from typing import List
import numpy as np
from skimage.color import rgb2gray
from sympy.physics.quantum.identitysearch import scipy
import scipy.signal
import math
import cv2
from scipy.ndimage import sobel
import scipy.ndimage as ndi


def normalize_tensor(img):
    # 计算每个通道的最大值和最小值
    Max = np.max(img, axis=0, keepdims=True)  # 沿着宽度W计算最大值
    Max = np.max(Max, axis=1, keepdims=True)  # 沿着高度H计算最大值
    Min = np.min(img, axis=0, keepdims=True)  # 沿着宽度W计算最小值
    Min = np.min(Min, axis=1, keepdims=True)  # 沿着高度H计算最小值

    img = (img - Min) / (Max - Min + 0.00005)
    return img

def entropy(image):
    """计算图像的熵 (Entropy, En)"""
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256), density=True)
    histogram += 1e-10  # 防止对数操作的零值
    en = -np.sum(histogram * np.log2(histogram))
    return en

def standard_deviation(image):
    """计算图像的标准差 (Standard Deviation, SD)"""
    mean_value = np.mean(image)
    sd = np.sqrt(np.mean((image - mean_value) ** 2))
    return sd

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, border: int = 0):
    img1 = normalize_tensor(img1)
    img2 = normalize_tensor(img2)
    if not img1.shape == img2.shape:
        img2 = img2[..., :img1.shape[-2], :img1.shape[-1]]
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def nabf(source1, source2, fused):
    """计算融合伪影 (Modified Fusion Artifacts Measure, Nabf)"""
    source1 = normalize_tensor(source1)
    source2 = normalize_tensor(source2)
    fused = normalize_tensor(fused)
    # 计算梯度
    gradient_source1 = np.hypot(sobel(source1, axis=0), sobel(source1, axis=1))
    gradient_source2 = np.hypot(sobel(source2, axis=0), sobel(source2, axis=1))
    gradient_fused = np.hypot(sobel(fused, axis=0), sobel(fused, axis=1))

    # 计算伪影度量
    diff1 = np.abs(gradient_fused - gradient_source1)
    diff2 = np.abs(gradient_fused - gradient_source2)
    nabf_value = np.sum(diff1 + diff2) / fused.size
    return nabf_value

def average_gradient(img):
    """
        计算图像的平均梯度。

        参数：
            img (np.ndarray): 输入图像，形状为 (H, W, C)，H 为高度，W 为宽度，C 为通道数。

        返回：
            avg_gradient (float): 图像的平均梯度。
        """
    # 计算水平方向和垂直方向的梯度
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # 水平方向
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # 垂直方向

    # 计算每个通道的梯度
    gradient_x = np.zeros_like(img)
    gradient_y = np.zeros_like(img)

    for c in range(img.shape[2]):  # 对每个通道分别处理
        gradient_x[:, :, c] = ndi.convolve(img[:, :, c], sobel_x, mode='reflect')
        gradient_y[:, :, c] = ndi.convolve(img[:, :, c], sobel_y, mode='reflect')

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # 计算平均梯度
    avg_gradient = np.mean(gradient_magnitude)

    return avg_gradient

def calculate_ssim(img1: np.ndarray, img2: np.ndarray, border: int = 0) -> float:
    def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))
        s: float = ssim_map.mean()
        return s

    img1 = normalize_tensor(img1)
    img2 = normalize_tensor(img2)
    if not img1.shape == img2.shape:
        img2 = img2[..., :img1.shape[-2], :img1.shape[-1]]
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims: List[float] = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')
    else:
        raise ValueError('Wrong input image dimensions.')

def correlation_coefficient(img1, img2):
    # img1 = normalize_tensor(img1)
    # img2 = normalize_tensor(img2)
    img1 = img1.flatten()
    img2 = img2.flatten()
    corr_coef = np.corrcoef(img1, img2)[0, 1]
    return corr_coef


def calculate_vif(ref, dist):
    def extract_features(image):
        # 转换为灰度图像
        gray = rgb2gray(image)

        # 计算局部均值和方差
        local_mean = scipy.signal.convolve2d(gray, np.ones((3, 3)) / 9, mode='same')
        local_var = scipy.signal.convolve2d(gray ** 2, np.ones((3, 3)) / 9, mode='same') - local_mean ** 2

        return gray, local_mean, local_var
    # ref = normalize_tensor(ref)
    # dist = normalize_tensor(dist)
    ref_gray, ref_mean, ref_var = extract_features(ref)
    dist_gray, dist_mean, dist_var = extract_features(dist)

    # 计算联合直方图
    bins = 256
    ref_hist, _ = np.histogram(ref_gray, bins=bins, density=True)
    dist_hist, _ = np.histogram(dist_gray, bins=bins, density=True)

    # 计算互信息
    joint_hist = np.outer(ref_hist, dist_hist)
    mutual_info = entropy(ref_hist) + entropy(dist_hist) - entropy(joint_hist.flatten())

    # 计算 VIF
    vif = 1 - np.exp(-mutual_info)

    return vif
