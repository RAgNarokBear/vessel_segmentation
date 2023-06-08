import torch
from torch.utils.data import DataLoader
import argparse
from data_util import VesselDataset
from model import VesselSeg


parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default='params_97.27acc.pth')
args = parser.parse_args()

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    dataset = VesselDataset(
        root_path='./data/CHASEDB1',
        label_mode='2',
    )

    test_loader = DataLoader(
        dataset=dataset[20:],
        batch_size=1,
        shuffle=False
    )

    model = VesselSeg()
    model.eval()
    print('load parameters:', args.c)
    model.load_state_dict(torch.load(args.c))
    model = model.to(device)

    tp, fp, fn, tn = 0, 0, 0, 0

    with torch.no_grad():
        for i, (test_x, test_y) in enumerate(test_loader):
            out = model(test_x.to(device))
            pred = torch.zeros_like(out)
            pred[out >= 0.5] = 1
            test_y = test_y.to(device)

            tp += ((pred == 1) & (test_y == 1)).int().sum()
            fp += ((pred == 1) & (test_y == 0)).int().sum()
            fn += ((pred == 0) & (test_y == 1)).int().sum()
            tn += ((pred == 0) & (test_y == 0)).int().sum()

    sp = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / (tp + fp + fn + tn)
    dice = 2 * tp / (fn + 2 * tp + fp)

    print('precision = {:.2%}'.format(precision))
    print('recall/se = {:.2%}'.format(recall))
    print('specificity = {:.2%}'.format(sp))
    print('f1 score = {:.2%}'.format(f1))
    print('accuracy = {:.2%}'.format(acc))
    print('DICE score = {:.2%}'.format(dice))


