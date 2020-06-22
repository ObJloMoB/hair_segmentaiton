import torch
import argparse

from lib.model import SegModel


def train(opts):
    model = SegModel()
    inp_t = torch.randn(1, 3, 512, 512)

    model(inp_t)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncl', help='number os classes',
                        default=2, type=int)
    args = parser.parse_args()
    train(args)
