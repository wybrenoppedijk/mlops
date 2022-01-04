import argparse
import sys

import torch
import numpy as np

from data import mnist
from model import MyAwesomeModel
import torch.nn.functional as F

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=2e-3)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        model = MyAwesomeModel().float()

        for epoch in range(10):
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            _, train_set = mnist()
            loss_avg = []
            for batch in train_set:
                x, y = batch
                y_hat = model(x.unsqueeze(1))

                # loss = F.cross_entropy(y_hat, y.long())
                loss = F.nll_loss(y_hat, y.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_avg.append(loss.data.item())
            print(np.mean(loss_avg))
            torch.save(model, "model.pt")

        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        model = torch.load(args.load_model_from)
        model.eval()
        _, test_set = mnist()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_set:
                x, y = batch
                y_hat = model(x.unsqueeze(1))
                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    TrainOREvaluate()
    
