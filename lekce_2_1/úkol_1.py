#!/usr/bin/env python3
# file: mnist_layers_activations.py
import argparse
import datetime
import re
import os
import statistics

# uncomment to run on CPU only
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time


"""
Přidal jsem si do tohoto skriptu svoje poznámky tak jen pro info
"""


print("Num GPUs Available: ", torch.cuda.device_count())

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--activation", default="none", choices=["none", "relu", "tanh", "sigmoid"], help="Activation.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--layers", default=1, type=int, help="Number of layers.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use. Zero value for default.")
# pokud bych chtěl měnit parametry tak to nedělat tady ale v Run configuration v pycharmu


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    # import random
    # random.seed(args.seed)
    # import numpy as np
    # np.random.seed(args.seed)

    accur = []

    for k in range(0,5):
        if args.threads > 0:
            torch.set_num_threads(args.threads)

        # Create logdir name
        logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
        ))

        # Load data
        mnist = MNIST()
        train_loader = DataLoader(mnist.train, batch_size=args.batch_size, shuffle=True)
        dev_loader = DataLoader(mnist.dev, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(mnist.test, batch_size=args.batch_size, shuffle=False)

        for X, y in train_loader:
            print(f"Shape and type of images is ([N, C, H, W]): {X.shape}, {X.dtype}")
            print(f"Shape and type of labels is : {y.shape}, {y.dtype}")
            break

        # Get cpu, gpu or mps device for training
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using {device} device")

        # Create the model
        model = nn.Sequential()


        # začátek cvičení


        # TODO: Finish the model. Namely add:
        # - a `nn.Flatten()` layer,
        # - `args.layers` number of `nn.Linear()` fully connected hidden layers with
        #   the same input size as the output size of the previous layer and the
        #   output size as the number of neurons there, i.e. `args.hidden_layer_size`,
        #   using activation from `args.activation`, allowing "none", "relu", "tanh", "sigmoid",
        # - finally, a final fully connected layer with
        #   `MNIST.LABELS` units and no activation (there is no need to use `nn.Softmax`, while using `nn.CrossEntropyLoss`)

        model.append(nn.Flatten())          # vytvořím novou vrstvu a přidám ji do modelu
        input_size = MNIST.C * MNIST.W *MNIST.H         # takhle bude velký ten vektor, jsou to vlastně rozměry obrázku
        output_size = args.hidden_layer_size

        for k in range(args.layers):        # vybírám jednotlivé vstupní funkce ze složky activation
            model.append(nn.Linear(input_size, output_size))    # připnu konkrétní funkci
            if args.activation=="none":
                print("Adding nonLinearity")
            elif args.activation=="relu":
                model.append(nn.ReLU())
                print("Adding layer with relu activation function")
            elif args.activation=="tanh":
                model.append(nn.Tanh())
                print("Adding layer with tanh activation function")
            elif args.activation=="sigmoid":
                model.append(nn.Sigmoid())
                print("adding layer with sigmoid activation function")
            else:
                print("Unknown activation")
                return -1
            input_size = args.hidden_layer_size     # aby se v druhém kroku korektně upravila velikost
        model.append(nn.Linear(input_size, MNIST.LABELS))   # výstupní vrstva


        # konec cvičení


        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('==================')
        print(f'Total parameters:{pytorch_total_params}')
        print(f'Trainable parameters:{pytorch_trainable_params}')
        print(f'Non-trainable parameters:{pytorch_total_params - pytorch_trainable_params}')
        print('==================')

        # Optimize the model
        # #RuntimeError: Windows not yet supported  for torch.compile
        # model.compile()

        model.to(device)

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Tensorboard writer initialization
        writer = SummaryWriter(logdir)

        # Training loop
        for epoch in range(args.epochs):
            print(f'Epoch {epoch + 1}/{args.epochs}:')
            # Training
            start_time = time.time()
            # Set the training mode flag, however here it is without any effect
            model.train()
            train_loss, train_correct = 0, 0
            num_batches = len(train_loader)
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()

                loss.backward()
                optimizer.step()
                batch_idx += 1
                # if batch_idx % 100 == 0:
                #     print(f"train loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

            train_acc = train_correct / len(train_loader.dataset)
            train_loss /= num_batches
            train_time = time.time() - start_time
            # print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_time: {train_time:.4f} ms')
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.flush()

            # Validation
            start_time = time.time()
            # Set the evaluation mode flag, however here it is without any effect
            model.eval()
            with torch.no_grad():
                val_loss, val_correct = 0, 0
                num_batches = len(dev_loader)
                for batch_idx, (images, labels) in enumerate(dev_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()

                    batch_idx += 1
                    # if batch_idx % 100 == 0:
                    #     print(f"val loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

            val_acc = val_correct / len(dev_loader.dataset)
            val_loss /= num_batches
            val_time = time.time() - start_time
            # print(f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_time: {val_time:.4f} s')
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Accuracy/validation", val_acc, epoch)
            writer.flush()

            print(
                f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_time: {train_time:.4f} s - val_loss: {val_loss:.2f} - val_acc: {val_acc:.4f} - val_time: {val_time:.4f} s')

        # Test
        start_time = time.time()
        # Set the evaluation mode flag, however here it is without any effect
        model.eval()
        with torch.no_grad():
            test_loss, test_correct = 0, 0
            num_batches = len(test_loader)
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()
                batch_idx += 1
                if batch_idx % 100 == 0:
                    print(f"test loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

        test_acc = test_correct / len(test_loader.dataset)
        test_loss /= num_batches
        test_time = time.time() - start_time
        print(f'test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f} - test_time: {test_time:.4f} s')
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        # appending accuary to the accur list
        accur.append(test_acc)
        print(accur)
        writer.flush()

        writer.close()

    # statistics
    result = (statistics.mean(accur), statistics.stdev(accur))

    return result


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    statistical_resolution = main(args)
    print(statistical_resolution)
    f = open("outputValue.txt", "w")
    f.write(f"{round(statistical_resolution[0], 6)}\n"
            f"{round(statistical_resolution[1], 6)}")

