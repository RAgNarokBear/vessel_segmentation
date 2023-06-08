import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from model import VesselSeg


def output_test():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    model = VesselSeg()
    model.eval()
    model.load_state_dict(torch.load('params_97.27acc.pth'))
    model = model.to(device)

    input_image = Image.open('data/CHASEDB1/Image_12L.jpg')
    test_out = Image.open('data/CHASEDB1/Image_12L_1stHO.png')
    W, H = input_image.size

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((480, 480)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.GaussianBlur(3)
    ])

    inverted_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((H, W)),
        torchvision.transforms.ToPILImage(),
    ])

    input_data = transform(input_image).unsqueeze(0).to(device)
    out_data = transform(test_out).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(input_data)
    pred = torch.zeros_like(out)
    pred[out >= 0.5] = 1

    print(((pred == 1) & (out_data == 1)).int().sum())

    out_image = inverted_transform(pred.squeeze(0))

    out_image.save('test_image.jpg')


def channel_test():
    input_image = Image.open('data/CHASEDB1/Image_12L.jpg')
    data = torch.tensor(np.array(input_image)).permute(2, 0, 1)
    r = data[0, :, :]
    g = data[1, :, :]
    b = data[2, :, :]

    min_gl, max_gl = b.min(), b.max()


    r_img = torchvision.transforms.ToPILImage()(r)
    g_img = torchvision.transforms.ToPILImage()(g)
    b_img = torchvision.transforms.ToPILImage()(b)
    b_img.show()
    b = 255 / (max_gl - min_gl) * (b - min_gl)
    b_img = torchvision.transforms.ToPILImage()(b)
    b_img.show()


    # r_img.show()
    # g_img.show()
    # b_img.show()
    # r_img.save('r.png')
    # g_img.save('g.png')
    # b_img.save('b.png')


if __name__ == '__main__':
    # output_test()
    channel_test()

    # a = np.array([1, 2, 3, 4])
    # plt.plot(a)
    # plt.show()