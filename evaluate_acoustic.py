import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from datasets.dataset import load_data
from models.model import *
from utils import *
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="ITRI_Small", type=str)
parser.add_argument('--abnormal_class', default="", type=str)
parser.add_argument("--data_type", default="acoustic", type=str)
parser.add_argument("--date", default="0922", type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--window_size', default=30, type=int)
parser.add_argument('--latent_dim', default=128, type=int)
parser.add_argument('--use_spatial', default=True, type=bool)
parser.add_argument('--times_std', default=10, type=int)
parser.add_argument('--ckpt_pth', default="./checkpoint/", type=str)
parser.add_argument('--figure_dir', default="./figures/", type=str)
parser.add_argument('--result_dir', default="./results/", type=str)
parser.add_argument('--num_samples', default=50, type=int)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

# Seed everything
seed_everything(seed=args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("> Using:", device)

dataset_name = args.dataset_name
batch_size = args.batch_size
img_size = args.img_size
latent_dim = args.latent_dim
use_spatial = args.use_spatial
num_samples = args.num_samples
times_std = args.times_std

ckp_path = args.ckpt_pth + dataset_name +'_3d_w{}.pth'.format(args.window_size) if use_spatial else args.ckpt_pth + dataset_name +'_2d.pth'
figure_path = "{figure_dir}/{use_spatial}/{abnormal_class}/".format(
    figure_dir=args.figure_dir, use_spatial="3d" if use_spatial else "2d", abnormal_class=args.abnormal_class)
result_path = args.result_dir + args.dataset_name + "/3d/" +'window_{window_size}/'.format(window_size=args.window_size) if use_spatial else args.result_dir + args.dataset_name + '/2d/'

os.makedirs(figure_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

''' Load testing data '''
train_dataloader, test_dataloader = load_data(dataset_name=dataset_name, args=args)

''' Autoencoder model '''
model = AutoEncoder_CNN(channel=3, dim=latent_dim, spatial=use_spatial).to(device)
model.load_state_dict(torch.load(ckp_path))
model.eval()

result = []

if use_spatial:
    info_path = "./jsons/{dataset}/3d/window_{window_size}/".format(
        dataset=dataset_name, window_size=args.window_size)
else:
    info_path = "./jsons/{dataset}/2d/".format(
        dataset=dataset_name)
os.makedirs(info_path, exist_ok=True)

info_file = info_path + "info.json"
''' 讀取 training error 的 mean 和 std '''
avg_raw_error, std_error = None, None
if os.path.exists(info_file):
    with open(info_file, "r") as f:
        train_status = json.load(f)
        print('-- Loading {}'.format(info_file))
        
        avg_raw_error = train_status["Average Raw Error"]
        std_error = train_status["std_error"]

# 如果找不到 training error 的 mean 和 std，就重新算一次
if avg_raw_error == None or std_error == None:
    print('-- Calculating training error mean and std')
    training_error = []
    for img, label in tqdm(train_dataloader):
        img = img.to(device)
        output = model(img)
        training_error.append(loss_function(img, output).item())

    avg_raw_error = np.mean(training_error)
    std_error = np.std(training_error)

    train_status = {
        "Dataset": "{dataset}_3d_window_{window_size}".format(dataset=dataset_name, window_size=args.window_size)
                if use_spatial else "{dataset}_2d".format(dataset=dataset_name),
        "Average Raw Error": avg_raw_error,
        "std_error": std_error,
    }
    with open(info_file, "w+") as f:
        json.dump(train_status, f, indent=4)
    print('-- Save training status to {}'.format(info_file))


avg_normal_error_1 = avg_raw_error + times_std*std_error
result.append({
    "Dataset": "{dataset}_3d_window_{window_size}".format(dataset=dataset_name, window_size=args.window_size)
            if use_spatial else "{dataset}_2d".format(dataset=dataset_name),
    "Normal Error + {}*std".format(times_std): avg_normal_error_1,
    "Average Raw Error": avg_raw_error,
    "std_error": std_error,
})


print("\n===== Training Data =====")
print('Average Normal Error: {:.4f}, Std: {:.4f}'.format(avg_raw_error, std_error))
print('Average Normal + {}*std Error: {:.4f}'.format(times_std, avg_normal_error_1))
print("=========================\n")


''' Evaluate on testing data '''
y_true = []
normal_errors, abnormal_errors, total_errors = [], [], []
normal_ssim, abnormal_ssim = [], []
with torch.no_grad():
    for idx, (img, label) in enumerate(tqdm(test_dataloader)):
        img = img.to(device)
        output = model(img)

        error = loss_function(img, output).item()
        total_errors.append(error)
        y_true.append(0 if label == 0 else 1)

        ''' Compute SSIM '''
        ssim_val = compute_ssim(img, output).mean().item()

        if label == 0:
            normal_errors.append(error)
            normal_ssim.append(ssim_val)
        else:
            abnormal_errors.append(error)
            abnormal_ssim.append(ssim_val)

        if idx < num_samples:
            item = {
                "ID": idx,
                "Label": label.item(),
                "Error": error,
                "Prediction": 1 if error > avg_normal_error_1 else 0
            }
            result.append(item)
            plot_result(img, output, "{}/{}.png".format(figure_path, idx), args)


normal_ssims = np.mean(normal_ssim)
abnormal_ssims = np.mean(abnormal_ssim)

y_pred_avg_1 = [1 if error > avg_normal_error_1 else 0 for error in total_errors]
acc_score = accuracy_score(y_true, y_pred_avg_1)
roc_score = roc_auc_score(y_true, total_errors)

result.append({
    "Accuracy [mean + {}*std]".format(times_std): acc_score,
    "Accuracy [roc]": roc_score,
    "SSIM [normal]": normal_ssims,
    "SSIM [abnormal]": abnormal_ssims,
})

print('Accuracy [mean + {}*std]: {:.4f}'.format(times_std, acc_score))
print('Accuracy [roc]: {:.4f}'.format(roc_score))
print('SSIM [normal]: {:.4f}'.format(normal_ssims))
print('SSIM [abnormal]: {:.4f}'.format(abnormal_ssims))


if use_spatial:
    json_path = "./jsons/{dataset}/3d/window_{window_size}/".format(
        dataset=dataset_name, window_size=args.window_size)    
else:
    json_path = "./jsons/{dataset}/2d/".format(
        dataset=dataset_name)
os.makedirs(json_path, exist_ok=True)

json_file = json_path + "result_{abnormal_class}.json".format(abnormal_class=args.abnormal_class)
with open(json_file, "w+") as f:
    json.dump(result, f, indent=4)
print('-- Save result to {}'.format(json_file))

''' Plot normal/abnormal error distribution '''
plot_distribution(normal_errors, abnormal_errors, avg_normal_error_1, result_path, args)