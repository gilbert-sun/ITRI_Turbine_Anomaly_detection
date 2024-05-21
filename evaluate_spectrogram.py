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
parser.add_argument("--dataset_name", default="ITRI_Big", type=str)
parser.add_argument('--abnormal_class', default="", type=str)
parser.add_argument("--data_type", default="spectrogram", type=str)
parser.add_argument("--date", default="0825", type=str)
parser.add_argument("--num_db", default=0, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--window_size', default=20, type=int)
parser.add_argument('--latent_dim', default=128, type=int)
parser.add_argument('--use_spatial', default=False, type=bool)
parser.add_argument('--times_std', default=1, type=float)
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

num_dbs = [0, 6, 12]
for db in num_dbs:
    args.num_db = db
    ckp_path = args.ckpt_pth + dataset_name + "_" + str(args.num_db) +'db_3d_w{}.pth'.format(args.window_size) if use_spatial else args.ckpt_pth + dataset_name + "_" + str(args.num_db) +'db_2d.pth'
    figure_path = "{figure_dir}/{dataset}/{use_spatial}/{db}db/".format(
        figure_dir=args.figure_dir, dataset=dataset_name, use_spatial="3d" if use_spatial else "2d", db=args.num_db)
    result_path = args.result_dir + args.dataset_name + "/" + str(args.num_db) + "db/3d/" +'window_{window_size}/'.format(window_size=args.window_size) if use_spatial else args.result_dir + args.dataset_name + "/" + str(args.num_db) + 'db/2d/'

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
        info_path = "./jsons/{dataset}/{db}db/3d/window_{window_size}/".format(
            dataset=dataset_name, db=args.num_db, window_size=args.window_size)
    else:
        info_path = "./jsons/{dataset}/{db}db/2d/".format(
            dataset=dataset_name, db=args.num_db)
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
            "Dataset": "{dataset}_{db}db_3d_window_{window_size}".format(dataset=dataset_name, db=args.num_db, window_size=args.window_size)
                    if use_spatial else "{dataset}_{db}db_2d".format(dataset=dataset_name, db=args.num_db),
            "Average Raw Error": avg_raw_error,
            "std_error": std_error,
        }
        with open(info_file, "w+") as f:
            json.dump(train_status, f, indent=4)
        print('-- Save training status to {}'.format(info_file))


    avg_normal_error_1 = avg_raw_error + times_std*std_error
    result.append({
        "Dataset": "{dataset}_{db}db_3d_window_{window_size}".format(dataset=dataset_name, db=args.num_db, window_size=args.window_size)
                if use_spatial else "{dataset}_{db}db_2d".format(dataset=dataset_name, db=args.num_db),
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
            
            ''' Calculate reconstruction error of a single file '''
            file_error = []
            img = img.squeeze(0)
            output_img = []
            for img_patch in img:
                img_patch = img_patch.unsqueeze(0)
                output = model(img_patch)
                error = loss_function(img_patch, output).item()
                file_error.append(error)
                output_img.append(output[0])  
            error = np.mean(file_error)

            total_errors.append(error)
            y_true.append(0 if label == 0 else 1)

            if label == 0:
                normal_errors.append(error)
            else:
                abnormal_errors.append(error)

            if idx < num_samples:
                item = {
                    "ID": idx,
                    "Label": label.item(),
                    "Error": error,
                    "Prediction": 1 if error > avg_normal_error_1 else 0
                }
                result.append(item)


    normal_ssims = np.mean(normal_ssim)
    abnormal_ssims = np.mean(abnormal_ssim)

    y_pred_avg_1 = [1 if error > avg_normal_error_1 else 0 for error in total_errors]
    acc_score = accuracy_score(y_true, y_pred_avg_1)
    roc_score = roc_auc_score(y_true, total_errors)

    result.append({
        "Accuracy [mean + {}*std]".format(times_std): acc_score,
        "Accuracy [roc]": roc_score,

    })

    print('Accuracy [mean + {}*std]: {:.4f}'.format(times_std, acc_score))
    print('Accuracy [roc]: {:.4f}'.format(roc_score))


    if use_spatial:
        json_path = "./jsons/{dataset}/{db}db/3d/window_{window_size}/".format(
            dataset=dataset_name, db=args.num_db, window_size=args.window_size)    
    else:
        json_path = "./jsons/{dataset}/{db}db/2d/".format(
            dataset=dataset_name, db=args.num_db)
    os.makedirs(json_path, exist_ok=True)

    json_file = json_path + "result.json"
    with open(json_file, "w+") as f:
        json.dump(result, f, indent=4)
    print('-- Save result to {}'.format(json_file))

    ''' Plot normal/abnormal error distribution '''
    plot_distribution(normal_errors, abnormal_errors, avg_normal_error_1, result_path, args)