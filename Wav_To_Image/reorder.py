import os
import shutil 

root_path = "./dataset/rawdata/"
save_path = "./dataset/wavset/"
date = "0922"
dir_path = os.path.join(root_path, date)

for dir_type in os.listdir(dir_path):
    folder_path = os.path.join(dir_path, dir_type)
    for db in os.listdir(folder_path):
        dest_path = os.path.join(save_path, date, db, dir_type)
        # os.makedirs(dest_path, exist_ok=True)

        src_path = os.path.join(folder_path, db)
        destination = shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        print("(src) {}, (dst) {}".format(src_path, dest_path))