#!/usr/bin/env python
"""
 @file   baseline.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1].
 @author Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2019 Hitachi, Ltd. All right reserved.
 [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," arXiv preprint arXiv:1909.09347, 2019.
"""
########################################################################
# import default python-library
########################################################################
import pickle
import os
import sys
import glob
import json
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.core
import librosa.feature
import yaml
import logging
# from import
from tqdm import tqdm
from sklearn import metrics
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import accuracy_score

matplotlib.use('Agg')

########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
########################################################################

# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=0):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : numpy.array( float )
        demuxed mono data

    Enabled to read multiple sampling rates.

    Enabled even one channel.
    """
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data

        return sr, numpy.array(multi_channel_data)[channel, :]

    except ValueError as msg:
        logger.warning(f'{msg}')


########################################################################


########################################################################
# feature extractor
########################################################################
def list_to_image(file_list,
                    msg="calc...",
                    n_mels=64,
                    frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0,
                    date="0922",
                    machine_type="ITRI",
                    machine_id="Small",
                    machine_db="0db",
                    data_type="normal",):

    file_path = "../data/spectrogram/{machine_type}_{machine_id}/{date}/{db}/{d_type}/".format(machine_type=machine_type, machine_id=machine_id, date=date, db=machine_db, d_type=data_type)
    os.makedirs(file_path, exist_ok=True)

    count = 0
    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):
        # make directory for each .wav file to save image slices
        file_name = file_list[idx]
        image_folder = file_path + file_name.split('/')[-1][:-4] + "/"
        os.makedirs(image_folder, exist_ok=True)

        sr, y = demux_wav(file_name)
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                        sr=sr,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        n_mels=n_mels,
                                                        power=power)
        
        # 04 calculate total vector size
        vectorarray_size = len(mel_spectrogram[0, :]) - frames + 1
        for i in range(0, vectorarray_size, 1):
            fig, ax = plt.subplots()
            # S_dB = librosa.power_to_db(mel_spectrogram[:, i: i+frames], ref=numpy.max)
            log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram[:, i: i+frames] + sys.float_info.epsilon)
            img = librosa.display.specshow(log_mel_spectrogram, vmin=-140, vmax=-40, x_axis='time',
                                        y_axis='mel', sr=sr,
                                        fmax=8000, ax=ax)
            save_name = image_folder + str(i) + ".png"
             # fig.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.axis('off')
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
            count += 1
            plt.close()
    return count



def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    sr, y = demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    
    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return numpy.empty((0, dims), float)

    # 06 generate feature vectors by concatenating multi_frames
    vectorarray = numpy.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    return vectorarray


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):

        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)

        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)

        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 

    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """
    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext))))
    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))
    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    # train_files = normal_files[len(abnormal_files):]
    # train_labels = normal_labels[len(abnormal_files):]
    # eval_files = numpy.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    # eval_labels = numpy.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    # logger.info("train_file num : {num}".format(num=len(train_files)))
    # logger.info("eval_file num : {num}".format(num=len(eval_files)))
    # return train_files, train_labels, eval_files, eval_labels

    logger.info("normal_file num : {num}".format(num=len(normal_files)))
    logger.info("abnormal_file num : {num}".format(num=len(abnormal_files)))

    return normal_files, abnormal_files


########################################################################


########################################################################
# main
########################################################################
if __name__ == "__main__":
    # load parameter yaml
    param = {
        "base_directory" : "./wavset",
        "date": "0825",
        "machine_type": "ITRI",
        "machine_id": "Big",
        "pickle_directory": "./pickle",
        "model_directory": "./model",
        "result_directory": "./result",
        "result_file": "result.yaml",
        "feature": {
            "n_mels": 64,
            "frames" : 5,
            "n_fft": 1024,
            "hop_length": 512,
            "power": 2.0,
        }
    }

    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/{date}/*".format(date=param["date"], base=param["base_directory"]))))
    machine_type = param["machine_type"]
    machine_id = param["machine_id"]

###############################################################################

    img_dataset_info = []
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

        # dataset param        
        # date = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        date = os.path.split(os.path.split(target_dir)[0])[1]
        db = os.path.split(target_dir)[1]

        # if db == "0db":
        #     continue

        normal_files, abnormal_files = dataset_generator(target_dir)
        normal_count = list_to_image(normal_files,
                                            msg="[Wav->Image] generate normal dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            date=date,
                                            machine_type=machine_type,
                                            machine_id=machine_id,
                                            machine_db=db,
                                            data_type="normal")

        abnormal_count = list_to_image(abnormal_files,
                                            msg="[Wav->Image] generate abnormal dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            date=date,
                                            machine_type=machine_type,
                                            machine_id=machine_id,
                                            machine_db=db,
                                            data_type="abnormal")

        info = {
            "dataset": "{}_{}_{}".format(machine_type, machine_id, db),
            "normal_data": normal_count,
            "abnormal_data": abnormal_count,
            "normal_files": len(normal_files),
            "abnormal_files": len(abnormal_files),
        }
        img_dataset_info.append(info)

    file_name = "./{type}_{id}_info.json".format(type=machine_type, id=machine_id)
    with open(file_name, "w") as f:
        json.dump(img_dataset_info, f, indent=4)
    print("save dataset_img_info to -> {}".format(file_name))