import time
import torch
import argparse
from utils import t_model
from test_dataset import LowLightDataset
from networks import LightenNet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='Hyper-parameters for LCENet')
parser.add_argument('--learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('--test_dir', help='Set the test_dir', default='C:/Users/Ryan/Documents/zzr/LOL/test/LOL/low/', type=str)
parser.add_argument('--save_dir', help='Set the save_dir', default='./out/', type=str)
parser.add_argument('--in_channels', help='Set the channels of input', default=3, type=int)
parser.add_argument('--out_channels', help='Set the channels of output', default=3, type=int)
parser.add_argument('--nf', help='Set the channels of network', default=32, type=int)
parser.add_argument('--batch_size', help='Set the test batch size', default=1, type=int)
parser.add_argument('--checkpoint_dir', help='directory for checkpoints', default='./checkpoints/', type=str)
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


t_transform = transforms.Compose([
    transforms.ToTensor(),
])


if __name__ == "__main__":
    t_data = LowLightDataset(low_dir=args.test_dir, transform=t_transform)
    t_loader = DataLoader(dataset=t_data, batch_size=args.batch_size, shuffle=False)
    net = LightenNet(args.in_channels, args.out_channels, args.nf)
    net = net.to(device)

    net.load_state_dict(torch.load(args.checkpoint_dir + 'best_model_parameters.pkl'))
    net.eval()
    print('--- Testing starts! ---')
    t_model(net, t_loader, device, args.save_dir)
