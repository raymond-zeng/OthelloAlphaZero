import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

from OthelloNNet import OthelloResNet

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


epochs = 1000
lr = 0.001
batch_size = 64

class NNetWrapper:
    def __init__(self):
        self.model = OthelloResNet(num_res_blocks=10, num_hidden_channels=128).cuda()

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(epochs):
            self.model.train()
            v_losses = AverageMeter()
            pi_losses = AverageMeter()
            batch_count = int(len(examples) / batch_size)
            for i in tqdm(range(batch_count), desc=f'Epoch {epoch + 1}/{epochs}'):
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs = zip(*[examples[sample_id] for sample_id in sample_ids])
                boards = torch.tensor(boards, dtype=torch.bfloat16).cuda()
                target_pis = torch.tensor(pis, dtype=torch.long).cuda()
                target_vs = torch.tensor(vs, dtype=torch.bfloat16).cuda()
                boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()
                out_pi, out_v = self.model(boards)
                pi_loss = loss_pi(target_pis, out_pi)
                v_loss = loss_v(target_vs, out_v)
                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.sum(param ** 2)
                loss = pi_loss + v_loss + 1e-4 * l2_reg
                pi_losses.update(pi_loss.item(), boards.size(0))
                v_losses.update(v_loss.item(), boards.size(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
def loss_pi(targets, outputs):
    return F.cross_entropy(outputs, targets)

def loss_v(targets, outputs):
    return F.mse_loss(outputs, targets)

def save_checkpoint(model, folder="./checkpoints", filename="checkpoint.pth.tar"):
    """
    Save the model checkpoint.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), f"{folder}/{filename}")
    print(f"Checkpoint saved to {folder}/{filename}")

def load_checkpoint(model, folder="./checkpoints", filename="checkpoint.pth.tar"):
    """
    Load the model checkpoint.
    """
    if os.path.exists(f"{folder}/{filename}"):
        model.load_state_dict(torch.load(f"{folder}/{filename}"))
        print(f"Checkpoint loaded from {folder}/{filename}")
    else:
        print(f"No checkpoint found at {folder}/{filename}")