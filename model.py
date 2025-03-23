import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import itertools
import time
import random

import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # Optimizes convolution operations

import helper_functions
from helper_functions import *


class CAN_data(Dataset):
    def __init__(self, data_file_name, labels, colnames):

        # labels = labels[:1000]
        
        self.data = np.memmap("model_data/"+data_file_name+".npy", dtype="float", mode="r", shape=(len(labels), 150, len(colnames)))
        self.labels = torch.tensor(np.stack(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float)
        y = self.labels[idx]
        return x,y
    

# ResNet block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=dilation*(kernel_size-1)//2, dilation=dilation),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, dilation=1),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x) 
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x) 
        out += residual
        out = self.relu(out)
        return out
    
class CANResNet(nn.Module):
    def __init__(self, block, nchannels, kernel_size, dilation, nlayers, fc_size, input_height, input_width, nclasses=5):
        super(CANResNet, self).__init__()
        self.nchannels = nchannels

        # Stem, only 1 layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=nchannels, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, dilation=1),
            nn.BatchNorm2d(num_features=nchannels)
        )

        height = (input_height - 1) // 2 + 1
        width  = (input_width  - 1) // 2 + 1

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        out_channels = nchannels
        in_channels  = nchannels
        nblocks = (nlayers - 1) // 2        # Every residual block has 2 layers in it

        for i in range(1, nblocks + 1):
            stride = 1
            if (nblocks > 1) and (i == (nblocks // 2 + 1)): # No striding if the number of layers is only 3 or 4
                stride = 2
                out_channels = out_channels * 2
                self.res_blocks.append(self._make_block(block, in_channels, out_channels, kernel_size, stride, dilation))
                in_channels = out_channels
            else:
                self.res_blocks.append(self._make_block(block, in_channels, out_channels, kernel_size, stride, dilation))
            height = (height - 1) // stride + 1
            width  = (width  - 1) // stride + 1
        
        # Extra convolution (if nlayers is even)
        self.extra_conv = None
        if ((nlayers - 1) % 2 == 1):        # Expression can be simplified, but this is easier to understand
            self.extra_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, dilation=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            )
  
        # height, width remain the same

        # Average pooling
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, count_include_pad=False)

        # Fully-connected layer
        self.fc1 = nn.Linear(in_features=out_channels * height * width, out_features=fc_size)
        self.fc2 = nn.Linear(in_features=fc_size, out_features=nclasses)


    def _make_block(self, block, in_channels, out_channels, kernel_size, stride, dilation):
        downsample = None
        if stride != 1 or in_channels != out_channels: 
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layer = block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, downsample=downsample)
        return layer
    
    def forward(self, x):
        out = self.stem(x)              # Stem
        for block in self.res_blocks:   # Residual blocks
            out = block(out)
        if self.extra_conv:
            out = self.extra_conv(out)  # Last layer, if present
        out = self.avgpool(out)         # Avg pool
        out = out.view(out.size(0), -1) # Reshape to 1D
        out = self.fc1(out)              # Fully-connected 
        out = self.fc2(out)
        return out

    

def train_CANResNet(config, trial=None):

    # Dataset
    with open('model_data/labels_train', 'rb') as fp:
        labels_train = pickle.load(fp)
    with open('model_data/labels_val', 'rb') as fp:
        labels_val = pickle.load(fp)
    with open('output/final_colnames_clustered', 'rb') as fp:    # Load column names and scaler
        colnames = pickle.load(fp)

    print("Loading datasets..")
    train_dataset = CAN_data(data_file_name="train_data", labels=labels_train, colnames=colnames)
    val_dataset = CAN_data(data_file_name="val_data", labels=labels_val, colnames=colnames)
    print("Datasets loaded.")

    # DataLoaders for training and validation datasets
    train_loader  = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    print("DataLoaders initialized.")

    # Initialize a model
    model = CANResNet(block=ResidualBlock,                  
                      nchannels=int(config['nchannels']), 
                      kernel_size=int(config['kernel_size']), 
                      dilation=int(config['dilation']), 
                      nlayers=int(config['nlayers']), 
                      fc_size=int(config['fc_size']),
                      input_height=150,
                      input_width=len(colnames),
                      nclasses=5) 
    print("Model initialized.")

    # Setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    model.to(device)  

    summary(model, (1, 150,  len(colnames)), batch_dim=0, verbose=1, col_names=["input_size","output_size","num_params"])

    # Loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config['lr'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        anneal_strategy='cos'
    )

    # Compile model
    # model = torch.compile(model, dynamic=True)

    scaler = torch.amp.GradScaler(device_str)

    val_loss    = 0.0
    val_steps   = 0
    total       = 0
    correct     = 0

    t_losses = []
    v_losses = []
    v_accs   = []

    for epoch in range(config['epochs']):
        model.train()                   
        running_loss = 0.0
        epoch_steps  = 0
        i = 0

        for i,data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{int(config["epochs"])}: ', unit="batch")):
            
            can_mats, labels = data 
            can_mats, labels = can_mats.to(device), labels.to(device)
            can_mats = can_mats.unsqueeze(1)

            optimizer.zero_grad()               # Zero gradients

            with torch.autocast(device_type=device_str):
                outputs = model(can_mats)           # Forward pass
                loss = criterion(outputs, labels)   # Compute loss
            
            # loss.backward()                     # Backward pass
            # optimizer.step()                    # Update weights
            # scheduler.step()                    # Update learning rate

            scaler.scale(loss).backward()       # Backward pass
            scaler.step(optimizer)              # Update weights

            s = scaler.get_scale()              # See https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/9
            scaler.update()
            if (s <= scaler.get_scale()):
                scheduler.step()                # Update learning rate

            running_loss += loss.item()         # Accumulate loss
            epoch_steps  += 1

            # if i % 2000 == 1999:
            #     print(f'Epoch {epoch+1} Batch {i+1} Loss: {running_loss / epoch_steps}')
            #     running_loss = 0.0
            # i += 1

        t_losses.append((running_loss / epoch_steps))

        # Validation loss
        model.eval()
        val_loss    = 0.0
        val_steps   = 0
        total       = 0
        correct     = 0

        # for can_mats, labels in val_loader:
        for i,data in enumerate(tqdm(val_loader, desc=f'    Validation: ', unit="batch")):
            with torch.no_grad():
                can_mats, labels = data 
                can_mats, labels = can_mats.to(device), labels.to(device)
                can_mats = can_mats.unsqueeze(1)    # Introducing channel dimension

                outputs = model(can_mats)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_steps += 1

        v_losses.append((val_loss / val_steps))
        v_accs.append((correct / total))

        print(f'    Training Loss: {running_loss / epoch_steps}, Validation Loss: {val_loss / val_steps}, Accuracy: {correct / total}')
        

    # Save model
    torch.save((model.state_dict(), optimizer.state_dict(), scaler.state_dict(), t_losses, v_losses, v_accs), "model_results/trial"+str(int(trial))+".pt")

    torch.cuda.empty_cache()
    print("Training completed.")
    print()


    return (running_loss / epoch_steps), (val_loss / val_steps), (correct / total)


def test_CANResNet(best_result, test_dataset, colnames, trial=None):

    # config for best model     
    config = best_result

    # Initialize best model
    model = CANResNet(block=ResidualBlock,                  
                      nchannels=int(config['nchannels']), 
                      kernel_size=int(config['kernel_size']), 
                      dilation=int(config['dilation']), 
                      nlayers=int(config['nlayers']), 
                      fc_size=int(config['fc_size']),
                      input_height=150,
                      input_width=len(colnames),
                      nclasses=5) 
    print("Best model initialized.")
    
    # Setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    model.to(device)  

    # Compile model
    # model = torch.compile(model, dynamic=True)

    # Load best model    
    checkpoint_path = "model_results/trial"+str(int(trial))+".pt"
    model_state, _, _, _, _, _ = torch.load(checkpoint_path)
    model.load_state_dict(model_state)
    print("Best model loaded from checkpoint.")

    # Test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=int(config['batch_size']), shuffle=False)

    # Testing
    model.eval()
    preds = []  # predicted labels
    actua = []  # actual labels
    total       = 0
    correct     = 0
    # for can_mats, labels in test_loader:
    for i, data in enumerate(tqdm(test_loader, desc=f"Test: ", unit="batch")):
        with torch.no_grad():
            can_mats, labels = data
            can_mats, labels = can_mats.to(device), labels.to(device)
            can_mats = can_mats.unsqueeze(1)
            
            with torch.autocast(device_type=device_str):
                outputs = model(can_mats)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            preds.append(predicted)
            actua.append(labels)

    print(f'Correct / Total: {correct} / {total}')
    print(f'Test Accuracy: {correct / total}')

    print("Saving actual and predicted labels to disk..")
    test_results = pd.DataFrame({
        'actual_label'      : torch.cat(actua).tolist(),
        'predicted_label'   : torch.cat(preds).tolist()
    })
    test_results.to_csv("model_results/test"+str(int(trial))+".csv", index=False)

    print("Testing completed.")
    print()


def hyperparameterSearch(config=None):

    # Generate all possible configurations 
    keys, values = zip(*config.items())
    config_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    keys=config_list[0].keys()

    if os.path.exists("model_results/search_results.csv"):
        search_results = pd.read_csv("model_results/search_results.csv")
        trial = int(search_results['trial'].max()) + 1
    else:
        search_results = pd.DataFrame(columns=[k for k in keys]+['time', 'train_loss', 'val_loss', 'accuracy', 'trial'])
        trial = 1

    if len(config_list) > 40:
        random.seed(101)
        sample_indices = random.sample(range(len(config_list)), 40)
        config_list = [config_list[i] for i in sample_indices]

    # config_list = [
    #     {
    #         'nchannels'     : 16,
    #         'kernel_size'   : 5,
    #         'dilation'      : 1,
    #         'nlayers'       : 3,
    #         'fc_size'       : 128,
    #         'lr'            : 1e-2,
    #         'weight_decay'  : 1e-5,
    #         'epochs'        : 20,
    #         'batch_size'    : 16384
    #     }
    # ]

    with open('model_results/config_list', 'wb') as fp:
        pickle.dump(config_list, fp)

    if trial > 1:
        config_list = config_list[(trial-1):]

    for conf in config_list:

        print("Trial", int(trial))
        print(conf)

        start_time = time.time()
        t_loss, v_loss, accuracy = train_CANResNet(config=conf, trial=trial)
        end_time   = time.time()

        c = conf.copy()

        c['time'] = end_time - start_time
        c['train_loss'] = t_loss
        c['val_loss'] = v_loss
        c['accuracy'] = accuracy
        c['trial'] = trial  

        search_results = pd.concat([search_results if not search_results.empty else None, pd.DataFrame([c])], ignore_index=True)

        print("Trial", int(trial), "complete.")

        trial = trial + 1

        print("Trials models sorted best to worst (by validation loss): ")
        print(search_results.sort_values(by='val_loss'))
        print()

        search_results.to_csv("model_results/search_results.csv", index=False)

        print("\nSaving best model configuration to disk..")
        best_result = dict(search_results.sort_values(by='val_loss').iloc[0])
        with open('model_results/best_result', 'wb') as fp:
            pickle.dump(best_result, fp)

    return best_result


def testBestModel(best_result=None):

    if best_result == None:
        with open('model_results/best_result', 'rb') as fp:
            best_result = pickle.load(fp)

    print("Testing best model..")

    print("Best hyperparameters found: ", best_result)

    # Dataset
    config=best_result
    with open('model_data/labels_test', 'rb') as fp:
        labels_test = pickle.load(fp)
    with open('output/final_colnames_clustered', 'rb') as fp:
        colnames = pickle.load(fp)

    test_dataset = CAN_data(data_file_name="test_data", labels=labels_test, colnames=colnames)
    test_CANResNet(best_result=best_result, test_dataset=test_dataset, colnames=colnames, trial=config['trial'])

# config = {
#     'nchannels'     : tune.choice([16, 32, 64, 128]),    
#     'kernel_size'   : tune.choice([5, 7, 9]),
#     'dilation'      : tune.choice([1, 2]),
#     'nlayers'       : tune.randint(3, 11),
#     'fc_size'       : tune.choice([128, 256, 512]),
#     'lr'            : tune.qloguniform(1e-4, 1e-2, 5e-5),
#     'weight_decay'  : tune.qloguniform(1e-5, 1e-3, 5e-6),
#     'epochs'        : 20,
#     'batch_size'    : 16384,
#     'file_path'     : '/home/bccc/shaila/'
# }

config = {
    'nchannels'     : [16, 32, 64],    
    'kernel_size'   : [5, 7, 9],
    'dilation'      : [1, 2],
    'nlayers'       : [3, 4, 5, 6, 7],
    'fc_size'       : [64, 128, 256],
    'lr'            : [1e-2],
    'weight_decay'  : [1e-5],
    'epochs'        : [15],
    'batch_size'    : [16384]
}

if __name__ == "__main__":  

    hyperparameterSearch(config=config)
    testBestModel()
