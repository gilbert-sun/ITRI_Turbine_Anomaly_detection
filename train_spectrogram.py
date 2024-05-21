import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from utils import *
from datasets.dataset import load_data
from models.model import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="ITRI_Big", type=str)
parser.add_argument('--abnormal_class', default="", type=str)
parser.add_argument("--data_type", default="spectrogram", type=str)
parser.add_argument("--date", default="0825", type=str)
parser.add_argument("--num_db", default=0, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--lr', default=0.003, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--window_size', default=20, type=int)
parser.add_argument('--latent_dim', default=128, type=int)
parser.add_argument('--use_spatial', default=False, type=bool)
parser.add_argument('--ckpt_pth', default="./checkpoint/", type=str)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

# Seed everything
seed_everything(seed=args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("> Using:", device)

dataset_name = args.dataset_name
num_epochs = args.num_epochs
lr = args.lr
batch_size = args.batch_size
img_size = args.img_size
latent_dim = args.latent_dim
use_spatial = args.use_spatial
ckpt_pth = args.ckpt_pth

os.makedirs(args.ckpt_pth, exist_ok=True)

num_dbs = [0, 6, 12]
for db in num_dbs:
    args.num_db = db
    ''' Load trainin data '''
    train_dataloader, test_dataloader = load_data(dataset_name=dataset_name, args=args)
    # save_image(make_grid(next(iter(train_dataloader))[:32][0]), dataset_name + "_normal.png")

    ''' Autoencoder model '''
    model = AutoEncoder_CNN(channel=3, dim=latent_dim, spatial=use_spatial).to(device)
    model.load_state_dict(torch.load(args.ckpt_pth + dataset_name + "_" + str(args.num_db) + "db_2d.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = 1
    for epoch in range(num_epochs):
        model.train()
        loss_list = []

        for idx, (img, label) in enumerate(tqdm(train_dataloader, leave=False)):
            img = img.to(device)
            output = model(img)

            loss = loss_function(img, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        avg_loss = np.mean(loss_list)
        if avg_loss <= best_loss:
            best_loss = avg_loss

            if use_spatial:
                ckpt_name = args.ckpt_pth + dataset_name + "_" + str(args.num_db) + "db_3d_w{window_size}.pth".format(window_size=args.window_size)
            else:
                ckpt_name = args.ckpt_pth + dataset_name + "_" + str(args.num_db) + "db_2d.pth"
            torch.save(model.state_dict(), ckpt_name)
            tqdm.write('{} | Epoch [{}/{}], loss:{:.4f}   -- save checkpoint'.format(dataset_name, epoch + 1, num_epochs, avg_loss))
        else:
            tqdm.write('{} | Epoch [{}/{}], loss:{:.4f}'.format(dataset_name, epoch + 1, num_epochs, avg_loss))