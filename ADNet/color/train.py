import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import time #tcw20182159tcw
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from torch.nn.modules.loss import _Loss #TCW20180913TCW
import matplotlib.pyplot as plt
from models import ADNet
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="ADNet")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--test_data", type=str, default='CBSD68', help='which test data to use')
parser.add_argument("--aug_train_data", type=bool, default=False, help='whether to augment the training data or not')
parser.add_argument("--noiseL", type=float, default=50, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=50, help='noise level used on validation set')
opt = parser.parse_args()

def plot(losses, psnrs):
    data_dict = {f"Train Loss SIDD": losses, f"Test PSNR {opt.test_data}": psnrs}

    fig, axs = plt.subplots(len(data_dict), figsize=(4,6), sharex=True)
    for i, item in enumerate(data_dict.items()):
        info, data = item
        s = info.split()
        split = s[0]
        metric = s[1]
        t_data = s[2]
        axs[i].plot(range(1, len(data)+1), data, label=metric)
        #axs[split].legend()
        axs[i].set_ylabel(metric)
        title = f"{split} {metric} on {t_data}"
        axs[i].set_title(title)
    axs[-1].set_xlabel("Epoch")
    fig.tight_layout()
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/aug-{opt.aug_train_data}_{opt.test_data}.png", bbox_inches="tight")

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def main():
    # Load dataset
    save_dir = opt.outf + 'sigma' + str(opt.noiseL) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = ADNet(channels=3, num_of_layers=opt.num_of_layers)
    #net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda() 
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    psnr_list = [] 
    loss_list = []
    for epoch in range(opt.epochs):
        if epoch <= opt.milestone:
            current_lr = opt.lr
        if epoch > opt.milestone and  epoch <=60:
            current_lr  =  opt.lr/10. 
        if epoch > 60  and  epoch <=90:
            current_lr = opt.lr/100.
        if epoch > 90:
            current_lr = opt.lr/1000.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        epoch_loss = 0
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.) 
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda()) 
            noise = Variable(noise.cuda())  
            out_train = model(imgn_train)
            loss =  criterion(out_train, img_train) / (imgn_train.size()[0]*2)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #model.eval()
            #out_train = torch.clamp(model(imgn_train), 0., 1.) 
            #psnr_train = batch_PSNR(out_train, img_train, 1.)
            #print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
            #    (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
        print("epoch %d loss: %.4f" % (epoch+1, epoch_loss/len(loader_train)))
        loss_list.append(epoch_loss/len(loader_train))
        model.eval() 
        # validate
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            torch.manual_seed(0) 
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda(),requires_grad=False)
            out_val = torch.clamp(model(imgn_val), 0., 1.)
            #print 'b'
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        psnr_list.append(psnr_val) 
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        model_name = 'model'+ '_' + str(epoch+1) + '.pth' 
        torch.save(model.state_dict(), os.path.join(save_dir, model_name)) 
    filename = save_dir + 'psnr.txt' 
    f = open(filename,'w') 
    for line in psnr_list: 
        f.write(f'{line}\n') 
    f.close()
    plot(loss_list, psnr_list)

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(opt, data_path='data', patch_size=50, stride=40, aug_times=1) 
        if opt.mode == 'B':
            prepare_data(opt, data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
