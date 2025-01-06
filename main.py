import os
import argparse
import numpy as np
import random
import torch
import torchvision
import yaml

import datasets
import models


def check_yml(d):
    try:
        assert os.path.isfile(d)
        return d
    except Exception:
        raise argparse.ArgumentTypeError(
            "yml file {} cannot be located.".format(d))

def main():
    parser = argparse.ArgumentParser(description='Project Name')
    parser.add_argument('mode', nargs="?",
                        choices=["train", "resume", "test", "visualize"])
    parser.add_argument('yml', nargs='?',
                        help='yml file name')

    args = parser.parse_args()

    check_yml(args.yml)
    with open(args.yml) as f:
        config = yaml.load(f, yaml.SafeLoader)

    # load param from yaml file
    use_cuda = not config['no_cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': os.cpu_count(), 'pin_memory': True} if use_cuda else {}

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # preprocessing 
    transform = torchvision.transforms.Compose([
        ])

    train_dataset = datasets.MockDataset(transform = transform)
    test_dataset = datasets.MockDataset(transform = transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['dataset']['train']['batch_size'], shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['dataset']['test']['batch_size'], shuffle=False, **kwargs)

    model = models.MockModel(device = device, train_loader = train_loader, test_loader = test_loader, config = config)

    if args.mode=='train':
        model.train()


    elif args.mode=='resume':
        model.load_weights()
        model.train()


    elif args.mode=='test':
        model.load_weights()
        model.test()


    elif args.mode=='visualize':
        model.load_weights()
        model.visualize()


if __name__ == '__main__':
    main()
