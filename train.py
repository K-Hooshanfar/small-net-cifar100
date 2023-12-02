# Import necessary libraries
import torch.nn as nn
import torch
import os
import sys
import argparse
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime
import torch.cuda
import matplotlib.pyplot as plt

# CIFAR-100 dataset mean and standard deviation for normalization
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Checkpoint path for saving model and logs
CHECK_POINT_PATH = "./checkpoint"

# Learning rate milestones for scheduler
MILESTONES = [60, 120, 160]


def training():
    # Set the model to train mode
    net.train()
    length = len(trainloader)
    total_sample = len(trainloader.dataset)
    total_loss = 0
    correct = 0

    # Iterate through batches in the training data
    for step, (x, y) in enumerate(trainloader):
        x = x.cuda()
        y = y.cuda()

        # Zero gradients, perform forward and backward passes, and update weights
        optimizer.zero_grad()
        output = net(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        # Update training statistics
        total_loss += loss.item()
        _, predict = torch.max(output, 1)
        correct += (predict == y).sum()

        # Write step information to the step log file
        fstep.write("Epoch:{}\t Step:{}\t TrainedSample:{}\t TotalSample:{}\t Loss:{:.3f}\n".format(
                epoch+1, step+1, step*args.b + len(y), total_sample, loss.item()
            ))
        fstep.flush()

        # Print training progress every 10 steps
        if step % 10 == 0:
            print("Epoch:{}\t Step:{}\t TrainedSample:{}\t TotalSample:{}\t Loss:{:.3f}".format(
                epoch+1, step+1, step*args.b + len(y), total_sample, loss.item()
            ))

    # Write epoch information to the epoch log file
    fepoch.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc:{:.3%}\n".format(
        epoch + 1, total_loss/length, optimizer.param_groups[0]['lr'], float(correct)/ total_sample
    ))
    fepoch.flush()
    return correct, total_sample, total_loss/length


def evaluating():
    # Set the model to evaluation mode
    net.eval()
    length = len(valloader)
    total_sample = len(valloader.dataset)
    total_loss = 0
    correct = 0

    # Iterate through batches in the validation data
    for step, (x, y) in enumerate(valloader):
        x = x.cuda()
        y = y.cuda()

        # Perform forward pass without gradient computation
        output = net(x)
        _, predict = torch.max(output, 1)
        torch.cuda.synchronize()
        loss = loss_function(output, y)
        total_loss += loss.item()
        correct += (predict == y).sum()

    # Calculate accuracy and write evaluation information to the eval log file
    acc = float(correct) / total_sample
    feval.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc:{:.3%}\n".format(
        epoch + 1, total_loss / length, optimizer.param_groups[0]['lr'], acc
    ))
    feval.flush()
    return acc, total_loss/length, total_loss


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", default='efficientnetb0', help='net type')
    parser.add_argument("-b", default=128, type=int, help='batch size')
    parser.add_argument("-lr", default=0.1, help='initial learning rate', type=int)
    parser.add_argument("-e", default=200, help='EPOCH', type=int)
    parser.add_argument("-optim", default="SGD", help='optimizer')
    args = parser.parse_args()

    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
    ])

    # Load CIFAR-100 dataset and split into train and validation subsets
    traindata = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(traindata))
    val_size = len(traindata) - train_size
    train_subset, val_subset = torch.utils.data.random_split(traindata, [train_size, val_size])
    trainloader = DataLoader(train_subset, batch_size=args.b, shuffle=True, num_workers=2)
    valloader = DataLoader(val_subset, batch_size=args.b, shuffle=False, num_workers=2)

    # Define neural network architecture
    if args.net == 'efficientnetb0':
        from models.efficientnet import efficientnet
        print("loading net")
        net = efficientnet(1, 1, 100, bn_momentum=0.9).cuda()
        print("loading finish")
    else:
        print('We don\'t support this net.')
        sys.exit()

    # Define loss, optimizer, learning rate scheduler, and checkpoint path
    print("defining training")
    loss_function = nn.CrossEntropyLoss()
    if args.optim == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2, last_epoch=-1)
    time = str(datetime.date.today() + datetime.timedelta(days=1))
    checkpoint_path = os.path.join(CHECK_POINT_PATH, args.net, time)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print("defining finish")

    # Train and evaluate the model
    best_acc = 0
    total_time = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    with open(os.path.join(checkpoint_path, 'EpochLog.txt'), 'w') as fepoch:
        with open(os.path.join(checkpoint_path, 'StepLog.txt'), 'w') as fstep:
            with open(os.path.join(checkpoint_path, 'EvalLog.txt'), 'w') as feval:
                with open(os.path.join(checkpoint_path, 'Best.txt'), 'w') as fbest:
                    print("start training")
                    for epoch in range(args.e):
                        correct, total_sample, averagelosstrain = training()
                        print("evaluating")
                        accuracy, averageloss, total_loss = evaluating()

                        # Append values for plotting
                        train_losses.append(averagelosstrain)
                        val_losses.append(averageloss)
                        train_accuracies.append(float(correct) / total_sample)
                        val_accuracies.append(accuracy)

                        scheduler.step()

                        print("saving regular")
                        torch.save(net.state_dict(), os.path.join(checkpoint_path, 'regularParam.pth'))

                        # if accuracy > best_acc:
                        print("saving best")
                        torch.save(net.state_dict(), os.path.join(checkpoint_path, 'bestParam.pth'))

                        fbest.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc:{:.3%}\n".format(
                                epoch + 1, averageloss, optimizer.param_groups[0]['lr'], accuracy
                            ))
                        fbest.flush()
                        best_acc = accuracy
    
    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_path, 'loss_plot.png'))

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_path, 'accuracy_plot.png'))

    plt.tight_layout()
    plt.show()
