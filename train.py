import copy

import torch
from torch.utils.data import DataLoader

from data_util import VesselDataset
from model import VesselSeg


def train(model, criterion, optimizer, train_data, test_data, total_epoch, device):
    # optimum model
    optimum_acc = 0
    optimum_model = None
    optimum_epoch = 0
    for epoch in range(1, total_epoch + 1):
        total_loss = 0
        for step, (train_x, train_y) in enumerate(train_data):
            out = model(train_x.to(device))
            loss = criterion(out, train_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # test and evaluate
        with torch.no_grad():
            matched, total = 0, 0
            for (test_x, test_y) in test_data:
                out = model(test_x.to(device))
                pred = torch.zeros_like(out)
                pred[out >= 0.5] = 1
                matched += (pred == test_y.to(device)).float().sum()
                total += test_x.reshape(3, -1).size(1)
            acc = matched / total
            if acc > optimum_acc:
                optimum_acc = acc
                optimum_epoch = epoch
                optimum_model = copy.deepcopy(model.state_dict())
        print('Epoch-{:02}  loss: {:.8f}  matched: {}  total: {}  acc: {:.2%}'.format(epoch, total_loss, matched, total, acc))
    return optimum_epoch, optimum_model, optimum_acc



if __name__ == '__main__':
    # args
    total_epoch = 50
    batch_size = 1
    learning_rate = 3e-4
    label_mode = '2'

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # data_loader
    train_dataset = VesselDataset(
        root_path='./data/CHASEDB1',
        label_mode=label_mode,
        # blue=True
    )

    test_dataset = VesselDataset(
        root_path='./data/CHASEDB1',
        label_mode=label_mode
    )

    train_loader = DataLoader(
        dataset=train_dataset[:20],
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset[20:],
        batch_size=batch_size,
        shuffle=False,
    )

    # model
    model = VesselSeg().to(device)

    # train
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch, params, acc = train(model, criterion, optimizer, train_loader, test_loader, total_epoch, device)

    # save
    print('Best acc: {:.2%}, {:02} epochs used.'.format(acc, epoch))
    torch.save(params, 'params_{:.2f}acc.pth'.format(acc * 100))

