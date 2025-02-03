

import os
import sys
import random
import pathlib
import time
import struct
import json
from array import array


import numpy as np

import torch as torch

from tqdm.notebook import tqdm

#from datasets import load_dataset, load_metric


PROJECT_FOLDER = pathlib.Path().resolve()
TRAIN_FOLDER = os.path.join(PROJECT_FOLDER, 'train_folder')
VALIDATION_FOLDER = os.path.join(PROJECT_FOLDER, 'validation_folder')
TEST_FOLDER = os.path.join(PROJECT_FOLDER, 'test_folder')
COMMON_FILES_FOLDER = os.path.join(PROJECT_FOLDER, 'common_files_folder')

LEARNING_RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, 'learning_results_1_folder')
if not os.path.isdir(LEARNING_RESULTS_FOLDER):
   os.makedirs(LEARNING_RESULTS_FOLDER)

GENERAL_FILE_TRAIN_PATH = os.path.join(TRAIN_FOLDER, 'general_file')
GENERAL_FILE_VALIDATION_PATH = os.path.join(VALIDATION_FOLDER, 'general_file')
GENERAL_FILE_TEST_PATH = os.path.join(TEST_FOLDER, 'general_file')


train_labels = {0: 2377,
 1: 2375,
 2: 2375,
 3: 2359,
 4: 2353,
 5: 2367,
 6: 2367,
 7: 2357,
 8: 2380,
 9: 2372,
 10: 6,
 11: 41039}
train_labels[10] += (94+60+60+59+60+59)   # because files of that label are longer than 1 second

weights = [sum(train_labels.values())/train_labels[label] for label in train_labels.keys()]



LONG_DATATYPE_BYTES = 4
DOUBLE_DATATYPE_BYTES = 8

NUMBER_OF_TRAIN_AUDIOTRACKS = int(os.path.getsize(GENERAL_FILE_TRAIN_PATH)/(5*LONG_DATATYPE_BYTES))
NUMBER_OF_VALIDATION_AUDIOTRACKS = int(os.path.getsize(GENERAL_FILE_VALIDATION_PATH)/(5*LONG_DATATYPE_BYTES))
NUMBER_OF_TEST_AUDIOTRACKS = int(os.path.getsize(GENERAL_FILE_TEST_PATH)/(5*LONG_DATATYPE_BYTES))

FOLDER_SIZE = 2048        # each folder contains spectrogramms of up to 2048 audiotracks

NUMBER_OF_TRAIN_FOLDERS = NUMBER_OF_TRAIN_AUDIOTRACKS // FOLDER_SIZE + 1
NUMBER_OF_VALIDATION_FOLDERS = NUMBER_OF_VALIDATION_AUDIOTRACKS // FOLDER_SIZE + 1
NUMBER_OF_TEST_FOLDERS = NUMBER_OF_TEST_AUDIOTRACKS // FOLDER_SIZE + 1

FRAME_LENGTH = 2048
HOP_LENGTH = 512
TIME_CUT_SIZE = 176 # this is number of frames equal to 4 seconds (1 sec = 44 frames)



class ConvModel(torch.nn.Module):
    def __init__(self, num_classes=12):
        super(ConvModel, self).__init__()

        self.conv_1 = torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=32, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.maxpool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = torch.nn.Conv2d(kernel_size=3, in_channels=32, out_channels=64, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.maxpool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=128, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.maxpool_3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = torch.nn.ReLU()

        self.lin_1 = torch.nn.Linear(in_features=128*128*5, out_features=256)
        self.lin_2 = torch.nn.Linear(in_features=256, out_features=num_classes)

        self.dropout_conv = torch.nn.Dropout(p=0.1)
        self.dropout_lin = torch.nn.Dropout(p=0.2)

    def forward(self, x):

        x = self.maxpool_1(self.relu(self.bn1(self.conv_1(x)))) # (batch_size, 1, 1024, 44) -> (batch_size, 32, 512, 22)
        x = self.dropout_conv(x)
        x = self.maxpool_2(self.relu(self.bn2(self.conv_2(x)))) # (batch_size, 32, 512, 22) -> (batch_size, 64, 256, 11)
        x = self.maxpool_3(self.relu(self.bn3(self.conv_3(x)))) # (batch_size, 64, 256, 11) -> (batch_size, 128, 128, 5)

        x = x.view(x.size(0), -1)  # (batch_size, 128, 128, 22) -> (batch_size, 128*128*22)

        x = self.relu(self.lin_1(x))  # (batch_size, 128*128*22) -> (batch_size, 256)
        x = self.dropout_lin(x)
        x = self.lin_2(x) # (batch_size, 256) -> (batch_size, 12)

        return x
    




class SoundDataset(torch.utils.data.Dataset):
    def __init__(self, spectrogramm_dataset_folder_path):

        self.spectrogramm_dataset_folder_path = spectrogramm_dataset_folder_path
        self.general_file_path = os.path.join(spectrogramm_dataset_folder_path, 'general_file')
        self.FOLDER_SIZE = 2048
        self.LONG_DATATYPE_BYTES = 4
        self.DOUBLE_DATATYPE_BYTES = 8

    def __getitem__(self, idx):
        
        # to read audiotrack
        # 1) choose right folder. Each folder contains spectrogramm of 2048 spectrogramms
        # so tracks 3,567,2047 will be in 1st folder, 2048,4095 will be in 2nd folder and so on
        # i-th folder name is os.path.join(TRAIN_FOLDER, f'spectrogramms_{i*FOLDER_SIZE}-{(i+1)*FOLDER_SIZE-1}')
        # 1) read file os.path.join(SUPERB_RESULTS_FOLDER, f'log_energy_spectrogramm_superb_v0_array_binary_i') where i is index of file (1st number)
        # 2) read chunks of bytes starting from byte given by 4th number - this is .seek() argument in spectrogramm file
        # 3) read doubles, which amount is equal to timeframes, and repeat this for each frequencyframe
        
        folder_index = idx//self.FOLDER_SIZE


        current_folder_path = os.path.join(self.spectrogramm_dataset_folder_path, f'{folder_index*self.FOLDER_SIZE}-{(folder_index+1)*self.FOLDER_SIZE-1}')
        current_file_path = os.path.join(current_folder_path, f'spectrogramm_{idx}')

        with open(current_file_path, 'rb') as infile_current:

            # read label first
            chunk = infile_current.read(self.LONG_DATATYPE_BYTES)
            label_of_current_track = struct.unpack("L", chunk)[0]
            label_of_current_track = torch.tensor(label_of_current_track)

            # Читаем форму спектрограммы
            shape_bytes = infile_current.read(8) # 8 байт для двух unsigned long
            shape = struct.unpack('LL', shape_bytes)

            #read spectrogramm
            spectrogram = np.fromfile(infile_current, dtype=np.float32).reshape(shape)
        
        spectrogram = torch.from_numpy(spectrogram.copy()).unsqueeze(0)

        return spectrogram, label_of_current_track

    def __len__(self):
        return int(os.path.getsize(self.general_file_path)/(5*LONG_DATATYPE_BYTES))


# TESTING DATASET WITHOUT 11-TH CLASS
with open(os.path.join(COMMON_FILES_FOLDER, 'indeces_without_11th_class.json'), 'r') as infile:
    filtered_indices = json.load(infile)

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, spectrogramm_dataset_folder_path):

        self.spectrogramm_dataset_folder_path = spectrogramm_dataset_folder_path
        self.general_file_path = os.path.join(spectrogramm_dataset_folder_path, 'general_file')
        self.FOLDER_SIZE = 2048
        self.LONG_DATATYPE_BYTES = 4
        self.DOUBLE_DATATYPE_BYTES = 8

    def __getitem__(self, idx):
        
        # to read audiotrack
        # 1) choose right folder. Each folder contains spectrogramm of 2048 spectrogramms
        # so tracks 3,567,2047 will be in 1st folder, 2048,4095 will be in 2nd folder and so on
        # i-th folder name is os.path.join(TRAIN_FOLDER, f'spectrogramms_{i*FOLDER_SIZE}-{(i+1)*FOLDER_SIZE-1}')
        # 1) read file os.path.join(SUPERB_RESULTS_FOLDER, f'log_energy_spectrogramm_superb_v0_array_binary_i') where i is index of file (1st number)
        # 2) read chunks of bytes starting from byte given by 4th number - this is .seek() argument in spectrogramm file
        # 3) read doubles, which amount is equal to timeframes, and repeat this for each frequencyframe
        idx = filtered_indices[idx]

        folder_index = idx//self.FOLDER_SIZE


        current_folder_path = os.path.join(self.spectrogramm_dataset_folder_path, f'{folder_index*self.FOLDER_SIZE}-{(folder_index+1)*self.FOLDER_SIZE-1}')
        current_file_path = os.path.join(current_folder_path, f'spectrogramm_{idx}')

        with open(current_file_path, 'rb') as infile_current:

            # read label first
            chunk = infile_current.read(self.LONG_DATATYPE_BYTES)
            label_of_current_track = struct.unpack("L", chunk)[0]
            label_of_current_track = torch.tensor(label_of_current_track)

            # Читаем форму спектрограммы
            shape_bytes = infile_current.read(8) # 8 байт для двух unsigned long
            shape = struct.unpack('LL', shape_bytes)

            #read spectrogramm
            spectrogram = np.fromfile(infile_current, dtype=np.float32).reshape(shape)
        
        spectrogram = torch.from_numpy(spectrogram.copy()).unsqueeze(0)

        return spectrogram, label_of_current_track

    def __len__(self):
        return len(filtered_indices)


def save_result(result, folder, name):
    current_file_path = os.path.join(folder, name)
    with open(current_file_path, mode="wb") as outfile_current:
        result = np.array(result)
        result.tofile(outfile_current)



filtered_dataset = FilteredDataset(spectrogramm_dataset_folder_path=TRAIN_FOLDER)
train_dataset = SoundDataset(spectrogramm_dataset_folder_path=TRAIN_FOLDER)
validation_dataset = SoundDataset(spectrogramm_dataset_folder_path=VALIDATION_FOLDER)
test_dataset = SoundDataset(spectrogramm_dataset_folder_path=TEST_FOLDER)





if __name__ == '__main__':
    device = torch.device("cuda")   #if torch.cuda.is_available() else "cpu"

    class_weights = torch.FloatTensor(weights).to(device)
    lr = 10**(-4)
    batch_size = 192
    num_epochs = 80

    model = ConvModel(num_classes=12)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn_train = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss_fn_test = torch.nn.CrossEntropyLoss()

    filtered_dataloader = torch.utils.data.DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)  #, num_workers=8, pin_memory=True
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    number_of_train_batches = len(train_dataset) // batch_size + 1
    number_of_test_batches = len(test_dataset) // batch_size + 1

    print_every_train = max(1, number_of_train_batches // 10) # Количество батчей для вывода (каждые 5%)
    print_every_test = max(1, number_of_test_batches // 10) 

    start_time = time.time()

    for i in range(num_epochs):
        
        train_loss = 0
        train_labels = []
        train_true_labels = []


        for j, (X, target) in enumerate(train_dataloader):
            
            X = X.to(device).float()
            target = target.to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss_value = loss_fn_train(preds, target)
            loss_value.backward()
            optimizer.step()

            train_loss = train_loss + loss_value.item()
            train_labels.extend(preds.argmax(axis=1).cpu().numpy().tolist())
            train_true_labels.extend(target.cpu().numpy().tolist())
            
            
            if (j+1) % print_every_train == 0:
                percent_complete = ((j+1) / number_of_train_batches) * 100
                print(f'Epoch: {i}, {percent_complete:.1f}%, Loss: {train_loss / j:.4f}, Time: {time.time() - start_time:.2f} sec')

            
            
        
        # simple metrics calculations
        train_loss /= number_of_train_batches
        train_accuracy = np.mean(np.array(train_labels) == np.array(train_true_labels))
        
        save_result(train_labels, folder=LEARNING_RESULTS_FOLDER, name=f'epoch_{i}_train_labels.npy')
        save_result(train_true_labels, folder=LEARNING_RESULTS_FOLDER, name=f'epoch_{i}_train_true_labels.npy')

        print(f'TRAIN: epoch = {i}, train_loss = {train_loss:.4f}, accuracy = {train_accuracy:.4f}, Time: {time.time() - start_time:.2f} sec')
        #print(f'train_labels[:192]\n{train_labels[:192]}')
        #print(f'train_true_labels[:192]\n{train_true_labels[:192]}')


        test_loss = 0
        test_labels = []
        test_true_labels = []

        with torch.no_grad():
            for j, (X, target) in enumerate(test_dataloader):
                X = X.to(device).float()
                target = target.to(device)
                
                preds = model(X)
                loss_value = loss_fn_test(preds, target)
                
                test_loss += loss_value.item()
                test_labels.extend(preds.argmax(axis=1).cpu().numpy().tolist())
                test_true_labels.extend(target.cpu().numpy().tolist())
                
                
                #if (j+1) % print_every_test == 0:
                    #percent_complete = ((j+1) / number_of_test_batches) * 100
                    #print(f'Epoch: {i}, {percent_complete:.1f}%, Loss: {test_loss / j:.4f}, Time: {time.time() - start_time:.2f} sec')
            
            test_loss /= number_of_test_batches
            test_accuracy = np.mean(np.array(test_labels) == np.array(test_true_labels))
        
            save_result(test_labels, folder=LEARNING_RESULTS_FOLDER, name=f'epoch_{i}_test_labels.npy')
            save_result(test_true_labels, folder=LEARNING_RESULTS_FOLDER, name=f'epoch_{i}_test_true_labels.npy')


            print(f'TEST: epoch =  {i}, test_loss = {test_loss:.4f}, accuracy = {test_accuracy:.4f}, Time: {time.time() - start_time:.2f} sec')
            #print(f'test_labels[:192]\n{test_labels[:192]}')
            #print(f'test_true_labels[:192]\n{test_true_labels[:192]}')