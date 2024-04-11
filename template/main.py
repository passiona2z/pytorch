
import dataset
from model import LeNet5, CustomMLP, LeNet5_improve
import seaborn as sns

# import some packages you need here
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training  # (tuple)
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value  
        acc: accuracy  
    """

    trn_loss_list = []
    trn_acc_list  = []
    tst_loss_list = []
    tst_acc_list  = []

    for epoch in range(40):

        model.train()  # 학습시작

        total_loss, total_samples, correct_num = 0,0,0

        for batch in trn_loader[0]:    
            image, label = map(lambda x: x.to(device), batch)
            predict = model(image)

            total_samples += len(predict)

            loss = criterion(predict, label)

            # 학습
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            
            # loss count
            total_loss    += loss.item()
            
            # accuracy count
            correct_num += (torch.argmax(predict, dim=1) == label).sum().item()


        trn_loss = total_loss / total_samples
        acc  = correct_num / total_samples

        trn_loss_list.append(trn_loss)
        trn_acc_list.append(acc)
       
        tst_loss, tst_acc = test(model, trn_loader[1], device, criterion)

        tst_loss_list.append(tst_loss)
        tst_acc_list.append(tst_acc)

        print(f"Epoch {epoch:2d}; trn_loss: {trn_loss:.3f}; trn_acc: {acc:.3f}| tst_loss: {tst_loss:.3f}; trn_acc: {tst_acc:.3f}")

    """
    # visualize
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sns.lineplot((trn_loss_list), marker='o', color='green', label='train')
    sns.lineplot(tst_loss_list, marker='o', color='red', label='test')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


    plt.subplot(1,2,2)
    sns.lineplot(trn_acc_list, marker='o', color='green', label='train')
    sns.lineplot(tst_acc_list, marker='o', color='red', label='test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')

    plt.tight_layout()
    plt.savefig('train.png')
    plt.show()
    """

    # model save
    torch.save(model, "model.pt")

    return trn_loss, acc



def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value  
        acc: accuracy  
    """ 

    # write your codes here

    model.eval()  

    total_loss, total_samples, correct_num = 0,0,0


    for batch in tst_loader:    
        image, label = map(lambda x: x.to(device), batch)
        predict = model(image)

        total_samples += len(predict)

        loss = criterion(predict, label)
        
        # loss count
        total_loss  += loss.item()
        
        # accuracy count
        correct_num += (torch.argmax(predict, dim=1) == label).sum().item()


        tst_loss = total_loss / total_samples
        acc  = correct_num / total_samples

    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    # config
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    
    # Dataset
    train_data = dataset.MNIST("../data/train")
    test_data  = dataset.MNIST("../data/test")

    # DataLoaders
    trn_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    tst_loader = DataLoader(test_data,  batch_size=batch_size)

    # model #LeNet5 # CustomMLP
    model = CustomMLP().to(device)
    print(model)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    
    # cost function
    criterion = nn.CrossEntropyLoss()

    # train
    trn_loss, trn_acc = train(model, (trn_loader, tst_loader), device, criterion, optimizer)

    # test
    tst_loss, tst_acc = test(torch.load("model.pt"), tst_loader, device, criterion)
    print(f"Test ; tst_loss: {tst_loss:.3f}; tst_acc: {tst_acc:.3f}")


if __name__ == '__main__':
    main()
