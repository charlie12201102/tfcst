import argparse

import cv2

import torch
from torchvision import transforms
import numpy as np
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from model.tfcos import TFCOSDetector


#torch.cuda.set_device(1)

#图像预处理——调整为指定大小、加padding
def preprocess_img(image, input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side = input_ksize
    h, w, _ = image.shape

    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w = 0
    pad_h = 0

    if nw % 16 != 0:
        pad_w = 16 - nw % 16
    if nh % 16 != 0:
        pad_h = 16 - nh % 16

    image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded


def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                             module.eps, module.momentum,
                                             module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name, convertSyncBNtoBN(child))
    del module
    return module_output


#得测试图像的检测结果——数据为序列图像
def main():
    #运行参数解析：包含模型权重路径、图像路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='checkpoint/rnn/TinyVovnet_ASFF_320_ConvLSTM-14_best.pth',help='Path to model checkpoint')
    parser.add_argument('--img_path', default='./test_images/list_img/Data1/',help='Path to images')

    opt = parser.parse_args()

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    #网络模型——TFCST
    model = TFCOSDetector(phase="test")
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[1]).cuda()
    else:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.model_path, map_location=torch.device('cpu')))

    print(opt.model_path)
    print(opt.img_path)

    model = model.eval()
    print("===>success loading model")

    import os

    root = opt.img_path
    names = os.listdir(root)
    print(names)
    #初始化TFCST中RNN单元体的状态变量
    state = [None] * 2
    for name in names:
        img_bgr = cv2.imread(root + name)
        img_pad = preprocess_img(img_bgr, [320, 320])
        img = cv2.cvtColor(img_pad.copy(), cv2.COLOR_BGR2RGB)
        img1 = transforms.ToTensor()(img)
        #归一化
        img1 = transforms.Normalize([0.42, 0.42, 0.42], [0.1, 0.1, 0.1], inplace=True)(img1)
        img1 = img1

        start_t = time.time()
        with torch.no_grad():
            out = model(img1.unsqueeze_(dim=0),state)
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing img, cost time %.2f ms" % cost_t)

        # 类别得分  类别  检测框 RNN的状态变量
        scores, classes, boxes,state = out

        boxes = boxes[0].cpu().numpy().tolist()
        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            img_pad = cv2.rectangle(img_pad, pt1, pt2, (0, 0, 255))
            # b_color = colors[int(classes[i]) - 1]
            b_color = colors[15]
            bbox = patches.Rectangle((box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], linewidth=1,
                                     facecolor='none', edgecolor='r')
            ax.add_patch(bbox)
            plt.text(box[0], box[1]-8, s="%.3f" % (scores[i]), color='white', fontsize=8,
                     verticalalignment='top',horizontalalignment='left',
                     bbox={'color': 'r', 'pad': 0})
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())


        name = name.split('.')[0]+'.png'

        plt.savefig('C:/Users/DELL/Desktop/out_images/{}'.format(name), bbox_inches='tight', pad_inches=0.0)
        plt.close()

        img_path='C:/Users/DELL/Desktop/out_images/{}'.format(name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        t = img_path[:-4]
        while(t[-2]=='0'):
            t=t[:-2]+t[-1]
        t = t + '.bmp'
        cv2.imwrite(t, img)

if __name__ == "__main__":
    main()