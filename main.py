import torch
import torch.nn as nn
from torchvision import transforms

import os
from train import train_model, get_train_val_dataloaders
from net import AlexNet


if __name__ == "__main__":
    import argparse
    import time
    curr_time = time.time()

    parser = argparse.ArgumentParser(description="Pass")

    default_save_path = f"checkpoints/model_{curr_time}.pth"

    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--save", type=str, default=default_save_path)
    parser.add_argument("--train", action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument("--resize", nargs="+", type=int, default=[70, 70])
    parser.add_argument("--random_crop", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--mean", nargs="+", type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument("--std", nargs="+", type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fraction_of_data", type=float, default=1.0)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--verbose", action='store_true')
    parser.set_defaults(verbose=True)
    parser.add_argument("--no-verbose", dest='verbose', action='store_false')

    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    print(f"Device: {DEVICE}")

    print("Initializing network...")
    alexnet = AlexNet(num_classes=10)

    if args.load_model_path:
        alexnet.load_state_dict(torch.load(args.load_model_path))
        print(f"Loaded AlexNet weights from locattion: {args.load_model_path}")
    else:
        print("Alexenet Network created!")
    if args.train:
        optimizer = torch.optim.SGD(alexnet.parameters(), momentum=0.9, lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, mode="max", verbose=True
        )
        print("Initialized optimizer, scheduler!")

        criterion = nn.CrossEntropyLoss()

        train_transforms = transforms.Compose(
            [
                transforms.Resize(args.resize),
                transforms.RandomCrop(args.random_crop),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ]
        )

        train_loader, val_loader = get_train_val_dataloaders(
            args.batch_size,
            args.fraction_of_data,
            args.val_size,
            train_transforms,
        )
        print("Dataloaders created")

        print("Checking datasets...")
        for images, labels in train_loader:
            print("Image batch dimensions:", images.shape)
            print("Image label dimensions:", labels.shape)
            print("Class labels of 10 examples:", labels[:10])
            break

        minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
            net=alexnet,
            num_epochs=args.num_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            device=DEVICE,
            verbose=args.verbose,
            log_freq=args.log_freq
        )
    if args.save:
        import os
        pwd = os.getcwd()
        save_path = os.path.join(pwd, args.save)
        torch.save(alexnet.state_dict(), save_path)
