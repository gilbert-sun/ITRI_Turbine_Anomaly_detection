import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_msssim import ssim
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from torchvision.transforms.functional import to_grayscale

def seed_everything(seed):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_function(a, b):
    mse_loss = torch.nn.MSELoss()
    return mse_loss(a, b)


def getImage(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            file_name = os.path.join(path, file)
            files.append(file_name)
    return files

def getFolder(path):
    files = []
    for file in os.listdir(path):
        file_name = os.path.join(path, file)
        files.append(file_name)
    return files


def invTrans(x):
    return (x + 1) / 2


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss


def plot_result(x, reconst, name, args):
    x, reconst = (x/2) + 0.5, (reconst/2) + 0.5

    if args.use_spatial:
        x, reconst = x[0].permute(0, 2, 3, 1)[::2], reconst[0].permute(0, 2, 3, 1)[::2]
        x, reconst = x.cpu().detach().numpy(), reconst.cpu().detach().numpy()
        error_map = []
        for i in range(len(x)):
            reconst_gray = cv2.cvtColor(reconst[i], cv2.COLOR_RGB2GRAY)
            x_gray = cv2.cvtColor(x[i], cv2.COLOR_RGB2GRAY)
            error = reconst_gray - x_gray
            error_map.append((error - error.min()) / (error.max() - error.min()))
        
        fig, ax = plt.subplots(3, 5, figsize=(5, 3))
        fig.tight_layout(pad=0)
        for i in range(5):
            ax[0][i].imshow(x[i], aspect='auto')
            ax[0][i].set_xticks([])
            ax[0][i].set_yticks([])
            # ax[0][i].set_axis_off()

            ax[1][i].imshow(reconst[i], aspect='auto')
            ax[1][i].set_xticks([])
            ax[1][i].set_yticks([])
            # ax[1][i].set_axis_off()
            ax[2][i].imshow(error_map[i], cmap="YlGnBu_r", vmax=1, vmin=0.6, aspect='auto')
            ax[2][i].set_xticks([])
            ax[2][i].set_yticks([])
            # ax[2][i].set_axis_off()
        # plt.show()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        x, reconst = x[0].permute(1, 2, 0), reconst[0].permute(1, 2, 0)
        x, reconst = x.cpu().detach().numpy(), reconst.cpu().detach().numpy()

        reconst_gray = cv2.cvtColor(reconst, cv2.COLOR_RGB2GRAY)
        x_gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        errors = reconst_gray - x_gray
        errors = (errors - errors.min()) / (errors.max() - errors.min())

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(x)
        ax[0].set_axis_off()
        ax[1].imshow(reconst)
        ax[1].set_axis_off()
        ax[2].imshow(errors, cmap="YlGnBu_r", vmax=1, vmin=0.6)
        ax[2].set_axis_off()

        # plt.axis('off')
        # plt.colorbar()
        # plt.imshow()
        # plt.clim(0.6, 1)
        plt.show()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_distribution(normal_errors, abnormal_errors, threshold, result_path, args):
    ''' Plot the distribution of reconstruction errors for normal and abnormal data '''
    if args.abnormal_class == "normal_out":
        args.abnormal_class = "Medium_Out"
    elif args.abnormal_class == "normal_in":
        args.abnormal_class = "Medium_In"
    elif args.abnormal_class == "low_in":
        args.abnormal_class = "Low_In"
    elif args.abnormal_class == "high_out":
        args.abnormal_class = "High_Out"
    elif args.abnormal_class == "normal":
        args.abnormal_class = "Normal State"

    title = 'Reconst. Error Distribution ({})'.format(args.abnormal_class)
    plt.figure(figsize=(10, 6))
    plt.hist(normal_errors, bins=50, density=True, alpha=0.5, label='Normal Data')
    plt.hist(abnormal_errors, bins=50, density=True, alpha=0.5, label='Abnormal Data')
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Mean + {} std'.format(args.times_std))
    plt.title(title)
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig("{}/result_{}.png".format(result_path, args.abnormal_class))
    # plt.savefig("{}/result_{}.png".format(result_path, args.abnormal_class), bbox_inches='tight', pad_inches=0)


def compute_ssim(x, reconst):
    x, reconst = (x[0] + 1) / 2, (reconst[0] + 1) / 2
    ssim_val = ssim(x, reconst, data_range=1, size_average=False, nonnegative_ssim=True)
    return ssim_val
