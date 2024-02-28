#!/usr/bin/env python3
# file: mnist_training.py
import argparse
import datetime
import re
import os

# uncomment to run on CPU only
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

print("Num GPUs Available: ", torch.cuda.device_count())

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--decay", default=None, choices=["linear", "exponential", "cosine"], help="Decay type.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=200, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Nesterov momentum to use in SGD.")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam"], help="Optimizer to use.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use. Zero value for default.")


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    # import random
    # random.seed(args.seed)
    # import numpy as np
    # np.random.seed(args.seed)

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

    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(MNIST.C * MNIST.H * MNIST.W, args.hidden_layer_size),
                          nn.ReLU(),
                          nn.Linear(args.hidden_layer_size, MNIST.LABELS),
                          # nn.Softmax(dim=1) # 0 dim=is for batch size, we do not need softmax, because we use nn.CrossEntropyLoss
                          )

    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('==================')
    print(f'Total parameters:{pytorch_total_params}')
    print(f'Trainable parameters:{pytorch_trainable_params}')
    print(f'Non-trainable parameters:{pytorch_total_params - pytorch_trainable_params}')
    print('==================')

    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    # TODO: Use the required `args.optimizer` (either `SGD` or `Adam`).
    # - For `SGD`, if `args.momentum` is specified, use Nesterov momentum.
    # - If `args.decay` is not specified, pass the given `args.learning_rate`
    #   directly to the optimizer as a `lr` argument.
    # - If `args.decay` is set, then
    #   - decay learning rate from args.learning_rate at the first training step
    #     downto args.learning_rate_final;
    #   - for `linear`, use `torch.optim.lr_scheduler.LinearLR` with the
    #     `start_factor=1.0`, and set `end_factor` and `total_iters` appropriately;
    #   - for `exponential`, use `torch.optim.lr_scheduler.ExponentialLR`,
    #     and set `gamma` appropriately;
    #   - for `cosine`, use `torch.optim.lr_scheduler.CosineAnnealingLR`,
    #     and set `T_max` and `eta_min` appropriately;
    #   - in all cases, you should reach the `args.learning_rate_final` just after the
    #     training, i.e., the first update after the training should use exactly the
    #     given `args.learning_rate_final`;
    #   - in all cases, `decay_steps` must be **the total number of optimizer updates**,
    #     i.e., the total number of training batches in all epochs;
    #   -  the total number of batches used for training in one epoch can be obtained by `len(train_loader)`
    #     returning an integer type;
    #   - call the created `{Polynomial,Exponential,CosineAnnealing}LR` scheduler `step`
    #      after each training step;
    #   - If a learning rate scheduler is used, you can call its `get_last_lr()`
    #      to obtain the last computed learning rate, however it returns a list.



    # optimizer excercixe
    if args.optimizer == "SGD":
        if args.momentum:
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                              momentum=args.momentum, nesterov=True)   # vyberu optimizer a řeknu mu paramtery co má optimalizovat
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    elif args.optimizer == "Adam":      # tady nastavím jiný optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        print("Unknown optimizer")      # vzhledem k tomu že tam josu jen 2 volby tak by to tady vubec nemuselo být to else
    # optimizer excercixe



    if args.decay:
        decay_rate_final = args.learning_rate_final/args.learning_rate   # nastavím si interval defacto
        decay_steps = len(train_loader) * args.epochs  # tady si nastavím počet iterací
        # len(train loader mi vrátí kolik batchi je v train loaderu a dělám to v každé epoše takže to tím vynásobím
        # uděláme si scheduler a dáme mu info že má pracovat s tím víše udělaným optimizerem
        # schedulery nejsou uplně unifikovaný takže se všude ty parametry jmenujou jinak no...
        if args.decay == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                          start_factor=1.0,
                                                          end_factor=decay_rate_final,
                                                          total_iters=decay_steps)
        elif args.decay == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                               gamma=np.power(decay_rate_final, 1/decay_steps))
                                                                                # chci si udělat velkou odmocninu
        elif args.decay == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=decay_steps,  # poslední krok
                                                                   eta_min=args.learning_rate_final)

    #Tensorboard writer initialization
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

            # tady je poslední krok cvičení, už je musíme ten vytvořený scheduler použít
            if args.decay:      # podmínka když scheduler existuje tak se protne
                scheduler.step()
            ...

            batch_idx +1
            if batch_idx % 100 == 0:
                #TODO: You can also print current learning rate here.
                # je docel aodbrý poku dho měním si ho taky vypsat
                ...
                print(f"train loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= num_batches
        train_time = time.time() - start_time
        # print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_time: {train_time:.4f} ms')
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        #TODO: Log current learning rate to TensorBoard at the end of each epoch.
        if args.decay:  # tady chceme tisknout learning rate ale scheduler nám to mění
            writer.add_scalar("Learning_rate/train", scheduler.get_last_lr()[0], epoch)
                                                    # je to seznam s hodnotami
        else:
            writer.add_scalar("Learning_rate/train", args.learning_rate, epoch)     # vypíše learning rate pro konkrétní epochu


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
                if batch_idx % 100 == 0:
                    print(f"val loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

        val_acc = val_correct / len(dev_loader.dataset)
        val_loss /= num_batches
        val_time = time.time() - start_time
        # print(f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_time: {val_time:.4f} s')
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)
        writer.flush()

        print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_time: {train_time:.4f} s - val_loss: {val_loss:.2f} - val_acc: {val_acc:.4f} - val_time: {val_time:.4f} s')

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
    writer.flush()

    writer.close()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
