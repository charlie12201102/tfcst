import collections

from torch import optim, nn

from dataset.DIY_dataset import DIYDataset
import torch
import math, time
from dataset.augment import Transforms

import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse

from model.tfcos import TFCOSDetector
from utils.callbacks import LossHistory

print('CUDA available: {}'.format(torch.cuda.is_available()))
#指定GPU卡号
torch.cuda.set_device(1)

#序列检测器的训练程序
def main(args=None):
    #运行命令中的参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--num_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument('--video_list', default='datasets/Indices/list/train_video_list.txt',
                        help='video name list')
    parser.add_argument('--pretrained_model_path', default='checkpoint/rnn/TinyVoVNet_320_ASFF_Pretian.pth',
                        help='single detector model path')
    parser.add_argument('--seq_len', type=int, default=14, help='number of list images')

    # parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    # parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    opt = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    
    #数据预处理
    transform = Transforms()
    #训练集加载程序
    train_dataset = DIYDataset(video_list=opt.video_list, resize_size=[320, 320], seq_len=opt.seq_len, skip=False,augment=transform)

    #网络模型——为论文中的TFCST网络
    model = TFCOSDetector(phase="train").cuda()
    model = torch.nn.DataParallel(model, device_ids=[1])

    #加载单帧检测器的预训练权重（单帧数据集的样本多样性更丰富）
    pretrained_net_dict = torch.load(opt.pretrained_model_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_net_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    #冻结特征提取和特征融合部分
    # freeze_nets = [model.module.fcos_body.backbone,model.module.fcos_body.fpn]
    # for freeze_net in freeze_nets:
    #     for param in freeze_net.parameters():
    #         param.requires_grad = False


    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs

    #数据集加载器  
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=opt.num_cpu, worker_init_fn=np.random.seed(0))

    print("total_videos : {}".format(len(train_dataset)))
    steps_per_epoch = len(train_dataset) // BATCH_SIZE


    #优化算法——Adam 初始学习率1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    # optimizer = optim.SGD(  # net.module.attention.parameters()
    #     [{'params': model.module.fcos_body.fpn.parameters()},
    #      {'params': model.module.fcos_body.head.parameters()},
    #      {'params': model.module.loss_layer.parameters()},
    #      {'params': model.module.target_layer.parameters()}]
    #     , lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    #
    # optimizer_rnn = optim.RMSprop(model.module.fcos_body.rnn_layer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True)

    # scheduler_rnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_rnn, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=50)

    model.train()

    #记录网络训练中的损失变化
    loss_history = LossHistory("logs/", "TinyVovnet_ASFF_320_ConvLSTM-14_lossRecord")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = []

        for iter_step, data in enumerate(train_loader):
            batch_imgs, batch_boxes, batch_classes = data
            batch_imgs = batch_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

            start_time = time.time()
            optimizer.zero_grad()
            
            #前向计算损失值
            losses = model([batch_imgs, batch_boxes, batch_classes])
            total_loss = losses[-1]
            
             #反向传播
            total_loss.mean().backward()


            # nn.utils.clip_grad_norm(model.module.fcos_body.rnn_layer.parameters(), 5)
            # optimizer_rnn.step()

            optimizer.step()

            loss_hist.append(float(total_loss))

            epoch_loss.append(float(total_loss))

            #运行时间
            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)
            
            #日志打印
            print("epoch:%d iters:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms total_loss:%.4f" % \
                  (epoch + 1, iter_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(), losses[2].mean(),
                   cost_time, np.mean(loss_hist)))

            if iter_step % 2 == 0:
                loss_history.append_loss_val(np.mean(loss_hist),losses[0].mean(),losses[1].mean(), losses[2].mean())

        print("lr:%.10f" % get_lr(optimizer))

        scheduler.step(np.mean(epoch_loss))
        # scheduler_rnn.step(np.mean(epoch_loss))

        #保存训练权重
        if epoch > EPOCHS-5 or epoch % 19==0:
            torch.save(model.state_dict(), "checkpoint/rnn/TinyVovnet_ASFF_320_ConvLSTM-14_F1_{}.pth".format(epoch + 1))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    main()
