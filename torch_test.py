

import os
import sys
import random
import pathlib
import time
import struct
from array import array


import numpy as np

import torch as torch


from tqdm.notebook import tqdm

#from datasets import load_dataset, load_metric


SOUND_EDA_FOLDER = pathlib.Path().resolve()

TRAIN_FOLDER = os.path.join(SOUND_EDA_FOLDER, 'train_folder')
if not os.path.isdir(TRAIN_FOLDER):
   os.makedirs(TRAIN_FOLDER)

VALIDATION_FOLDER = os.path.join(SOUND_EDA_FOLDER, 'validation_folder')
if not os.path.isdir(VALIDATION_FOLDER):
   os.makedirs(VALIDATION_FOLDER)

TEST_FOLDER = os.path.join(SOUND_EDA_FOLDER, 'test_folder')
if not os.path.isdir(TEST_FOLDER):
   os.makedirs(TEST_FOLDER)

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

weights = [sum(train_labels.values())/train_labels[label] for label in train_labels.keys()]



NUMBER_OF_TRAIN_AUDIOTRACKS = 64727
NUMBER_OF_VALIDATION_AUDIOTRACKS = 6798
NUMBER_OF_TEST_AUDIOTRACKS = 3081

FOLDER_SIZE = 2048        # each folder contains spectrogramms of up to 2048 audiotracks

NUMBER_OF_TRAIN_FOLDERS = NUMBER_OF_TRAIN_AUDIOTRACKS // FOLDER_SIZE + 1
NUMBER_OF_VALIDATION_FOLDERS = NUMBER_OF_VALIDATION_AUDIOTRACKS // FOLDER_SIZE + 1
NUMBER_OF_TEST_FOLDERS = NUMBER_OF_TEST_AUDIOTRACKS // FOLDER_SIZE + 1

FRAME_LENGTH = 2048
HOP_LENGTH = 512
TIME_CUT_SIZE = 176 # this is number of frames equal to 4 seconds (1 sec = 44 frames)

LONG_DATATYPE_BYTES = 4

DOUBLE_DATATYPE_BYTES = 8




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

        self.lin_1 = torch.nn.Linear(in_features=128*128*22, out_features=256)
        self.lin_2 = torch.nn.Linear(in_features=256, out_features=num_classes)

        self.dropout_conv = torch.nn.Dropout(p=0.1)
        self.dropout_lin = torch.nn.Dropout(p=0.2)

    def forward(self, x):

        x = self.maxpool_1(self.relu(self.bn1(self.conv_1(x)))) # (batch_size, 1, 1024, 176) -> (batch_size, 32, 512, 88)
        x = self.dropout_conv(x)
        x = self.maxpool_2(self.relu(self.bn2(self.conv_2(x)))) # (batch_size, 32, 512, 88) -> (batch_size, 64, 256, 44)
        x = self.maxpool_3(self.relu(self.bn3(self.conv_3(x)))) # (batch_size, 64, 256, 44) -> (batch_size, 128, 128, 22)

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


        current_folder_path = os.path.join(self.spectrogramm_dataset_folder_path, f'spectrogramms_{folder_index*self.FOLDER_SIZE}-{(folder_index+1)*self.FOLDER_SIZE-1}')
        current_file_path = os.path.join(current_folder_path, f'log_energy_spectrogramm_superb_array_binary_{idx}')

        with open(current_file_path, 'rb') as infile_current:

            # read label first
            chunk = infile_current.read(self.LONG_DATATYPE_BYTES)
            label_of_current_track = struct.unpack("L", chunk)[0]
            label_of_current_track = torch.tensor(label_of_current_track)


            #read spectrogramm
            data_bytes = infile_current.read()

        spectrogram = np.frombuffer(data_bytes, dtype=np.float64).reshape(1025, 176)
        spectrogram = torch.from_numpy(spectrogram.copy()).unsqueeze(0)
        
        return spectrogram, label_of_current_track

    def __len__(self):
        return int(os.path.getsize(self.general_file_path)/(5*LONG_DATATYPE_BYTES))





train_dataset = SoundDataset(spectrogramm_dataset_folder_path=TRAIN_FOLDER)
validation_dataset = SoundDataset(spectrogramm_dataset_folder_path=VALIDATION_FOLDER)
test_dataset = SoundDataset(spectrogramm_dataset_folder_path=TEST_FOLDER)





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = torch.FloatTensor(weights).to(device)
    lr = 10**(-3)
    batch_size = 64
    num_epochs = 100

    model = ConvModel()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)  #, num_workers=8, pin_memory=True
    #validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    number_of_train_batches = len(train_dataset) // batch_size + 1
    number_of_test_batches = len(test_dataset) // batch_size + 1

    start_time = time.time()

    for i in tqdm(range(num_epochs)):
        
        train_loss = 0
        train_labels = []
        train_true_labels = []

        counter = 0

        for X, target in tqdm(train_dataloader):
            
            X = X.to(device).float()
            target = target.to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss_value = loss_fn(preds, target)
            loss_value.backward()
            optimizer.step()

            train_loss = train_loss + loss_value.item()
            train_labels.extend(preds.argmax(axis=1).cpu().numpy().tolist())
            train_true_labels.extend(target.cpu().numpy().tolist())
            
            counter += 1
            if counter%(number_of_train_batches//100+1) == 0:
                print(f'{counter//(number_of_train_batches//100+1)} %, time = {time.time() - start_time}, counter = {counter}')
            
            
        
        # simple metrics calculations
        train_loss /= len(train_dataloader)
        train_accuracy = np.mean(np.array(train_labels) == np.array(train_true_labels))
        
        print(f'TRAIN: epoch = {i}, train_loss = {train_loss}, accuracy = {train_accuracy}')


        test_loss = 0
        test_labels = []
        test_true_labels = []
            
        counter = 0

        with torch.no_grad():
            for X, target in test_dataloader:
                X = X.to(device).float()
                target = target.to(device)
                
                preds = model(X)
                loss_value = loss_fn(preds, target)
                
                test_loss += loss_value.item()
                test_labels.extend(preds.argmax(axis=1).cpu().numpy().tolist())
                test_true_labels.extend(target.cpu().numpy().tolist())
                
                counter += 1
                if counter%(number_of_test_batches//10+1) == 0:
                    print(f'{ (counter//(number_of_test_batches//10+1))*10 } %, time = {time.time() - start_time}, counter = {counter}')
            
            test_loss /= len(test_dataloader)
            test_accuracy = np.mean(np.array(test_labels) == np.array(test_true_labels))
        
        


            print(f'TRAIN: epoch =  {i}, test_loss = {test_loss}, accuracy = {test_accuracy}')