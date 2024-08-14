import torch
from torchvision.models import vgg11,resnet18,resnet101,resnext101_32x8d
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.dataloader.samplers import CategoriesSampler
from Models.utils import *
from Models.dataloader.data_utils import *
# from Models.models.Network import DeepEMD
# from Models.models.yuanNetwork import DeepEMD
from Models.models.attNetwork import DeepEMD
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# from Models.models.renet import DeepEMD
from PIL import Image

from torch.autograd import Variable
import utils
from torch.cuda.amp import autocast as autocast
# from pytorch_model_summary import summary

from torch.utils.tensorboard import SummaryWriter
import tqdm
import time
PRETRAIN_DIR='deepemd_pretrain_model/'
# DATA_DIR='/home/zhangchi/dataset'
DATA_DIR=r'F:/few-shot dataset'
# DATA_DIR='/mnt/f/few-shot dataset'

from thop import profile
parser = argparse.ArgumentParser()
#about dataset and training
parser.add_argument('-dataset', type=str, default='cifar_fs', choices=['miniimagenet', 'cub','tieredimagenet','fc100','tieredimagenet','cifar_fs'])
parser.add_argument('-data_dir', type=str, default=DATA_DIR,help='dir of datasets')
parser.add_argument('-set',type=str,default='test',choices=['test','val'],help='the set used for validation')# set used for validation
#about training
parser.add_argument('-bs', type=int, default=1,help='batch size of tasks')
parser.add_argument('-max_epoch', type=int, default=100)
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-step_size', type=int, default=10)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument('-val_frequency',type=int,default=50)
parser.add_argument('-random_val_task',action='store_true',help='random samples tasks for validation at each epoch')
parser.add_argument('-save_all',default=True,action='store_true',help='save models on each epoch')
#about task
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=5)
parser.add_argument('-query', type=int, default=15,help='number of query image per class')
parser.add_argument('-val_episode', type=int, default=500, help='number of validation episode')
parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')
# about model x
parser.add_argument('-pretrain_dir', type=str, default=PRETRAIN_DIR)
parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
parser.add_argument('-norm', type=str, default='center', choices=['center'], help='feature normalization')
parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
#deepemd fcn only
parser.add_argument('-feature_pyramid', type=str, default=None, help='you can set it like: 2,3')
#deepemd sampling only
parser.add_argument('-num_patch',type=int,default=9)
#deepemd grid only patch_list
parser.add_argument('-patch_list',type=str,default='2,3',help='the size of grids at every image-pyramid level')
parser.add_argument('-patch_ratio',type=float,default=2,help='scale the patch to incorporate context around the patch')
# slvoer about
parser.add_argument('-solver', type=str, default='opencv', choices=['opencv', 'qpth'])
parser.add_argument('-form', type=str, default='L2', choices=['QP', 'L2'])
parser.add_argument('-l2_strength', type=float, default=0.000001)
# SFC
parser.add_argument('-sfc_lr', type=float, default=0.1, help='learning rate of SFC')
parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
parser.add_argument('-sfc_update_step', type=float, default=100, help='number of updating step of SFC')
parser.add_argument('-sfc_bs', type=int, default=4, help='batch size for finetune sfc')
parser.add_argument('-temperature2', type=float, default=1.0)
parser.add_argument('-alpha', type=float, default=0.7, help='the balanced parameters between loss function')

# OTHERS
parser.add_argument('-gpu', default='0')
parser.add_argument('-extra_dir', type=str,default=None,help='extra information that is added to checkpoint dir, e.g. hyperparameters')
parser.add_argument('-seed', type=int, default=1)
# parser.add_argument('-pretrain_dir', type=str, default=PRETRAIN_DIR)
# parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
# parser.add_argument('-norm', type=str, default='center', choices=['center'], help='feature normalization')
# parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
parser.add_argument('-N', type=int, default=3, help='the deep of QSFormer')
parser.add_argument('-head', type=int, default=10, help='the head number of multi-head attention in cross_feature')
parser.add_argument('-head_metric', type=int, default=1, help='the head number of multi-head attention in sampleFormer')
parser.add_argument('-head_enc', type=int, default=8, help='the head number of multi-head attention in patchFormer')
parser.add_argument('-dp1', type=float, default=0.5, help='the set of dropout in cross_feature')
parser.add_argument('-dp2', type=float, default=0.5, help='the set of dropout in the encoder of QS-Decoder')
parser.add_argument('-dp3', type=float, default=0.5, help='the set of dropout in the SA of QS-Decoder')
parser.add_argument('-dp4', type=float, default=0.1, help='the set of dropout in patchFormer')

parser.add_argument('-tau', type=float, default=0.7, help='the parameters of constractive_loss')
parser.add_argument('-lamda1', type=float, default=0.9, help='(a) the balanced parameters of feature in patchFormer')
parser.add_argument('-lamda2', type=float, default=0.9, help='the balanced parameters of feature in sampleFormer')
parser.add_argument('-lamda', type=float, default=0.1, help='the balanced parameters of similarity between patchFormer and sampleFormer')

args = parser.parse_args()
pprint(vars(args))


def keshihua(model,img,imput):
    target_layers = [model.module.agent]
    # 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
    cam = GradCAM(model=model, target_layers=target_layers)
    # targets = [ClassifierOutputTarget(preds)]
    # 上方preds需要设定，比如ImageNet有1000类，这里可以设为200
    grayscale_cam = cam(input_tensor=imput)
    # grayscale_cam = grayscale_cam[0, :]
    cam_img = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    print(type(cam_img))
    Image.fromarray(cam_img)

# 定义一个全局容器来保存特征图
activation = {}

# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook



#transform str parameter into list
if args.feature_pyramid is not None:
    args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
args.patch_list = [int(x) for x in args.patch_list.split(',')]

set_seed(args.seed)
num_gpu = set_gpu(args)
Dataset=set_up_datasets(args)

# model
args.pretrain_dir=osp.join(args.pretrain_dir,'%s/resnet12/max_acc.pth'%(args.dataset))
model = DeepEMD(args)
model = load_model(model, args.pretrain_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()
# model = vgg11(pretrained=True)
img_path = './dog.jpg'
# resize操作是为了和传入神经网络训练图片大小一致
img = Image.open(img_path).resize((224,224))
# 需要将原始图片转为np.float32格式并且在0-1之间
rgb_img = np.float32(img)/255
rgb_img = rgb_img[:,:,:3]
plt.imshow(img)
preds = 200
from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 将图片转为tensor




args.save_path = '%s/%s/%dshot-%dway/'%(args.dataset,args.deepemd,args.shot,args.way)

args.save_path=osp.join('checkpoint',args.save_path)
if args.extra_dir is not None:
    args.save_path=osp.join(args.save_path,args.extra_dir)
ensure_path(args.save_path)


trainset = Dataset('train', args)
train_sampler = CategoriesSampler(trainset.label, args.val_frequency*args.bs, args.way, args.shot + args.query)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)

valset = Dataset(args.set, args)
val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=0, pin_memory=True)

if not args.random_val_task:
    print ('fix val set for all epochs')
    # val_loader=[x for x in val_loader]
print('save all checkpoint models:', (args.save_all is True))

#label for query set, always in the same pattern
label = torch.arange(args.way, dtype=torch.int8).repeat(args.query)#012340123401234...
label = label.type(torch.LongTensor)
label = label.cuda()



optimizer = torch.optim.SGD([{'params': model.parameters(),'lr':args.lr}], momentum=0.9, nesterov=True, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
pytorch_total_params = sum(p.numel() for p in model.parameters())
trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total - ', pytorch_total_params)
print('Trainable - ', trainable_pytorch_total_params)
with autocast():
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))



    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    global_count = 0
    writer = SummaryWriter(osp.join(args.save_path,'tf'))

    result_list=[args.save_path]
    for epoch in range(1, args.max_epoch + 1):
        print (args.save_path)
        start_time=time.time()

        tl = Averager()
        ta = Averager()


        tqdm_gen = tqdm.tqdm(train_loader)
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm_gen, 1):

            global_count = global_count + 1
            data, _ = [_.cuda() for _ in batch]

            k = args.way * args.shot
            model.module.mode = 'encoder'
            data = model(data)


            data_shot, data_query = data[:k], data[k:]
            model.module.mode = 'meta'
            if args.shot > 1:
                data_shot = model.module.get_sfc(data_shot)
            # logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
            data = torch.cat( (data_shot, data_query),dim=0)

            # logits, logits_trans = model(data)
            img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0)

            target_layers = [model.module.agent]
            # 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
            cam = GradCAM(model=model, target_layers=target_layers)
            # targets = [ClassifierOutputTarget(preds)]
            # 上方preds需要设定，比如ImageNet有1000类，这里可以设为200
            # data = torch.cat( (data_shot, data_query),dim=0)
            grayscale_cam = cam(input_tensor=data)
            grayscale_cam = grayscale_cam[0, :]
            cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            print(type(cam_img))
            Image.fromarray(cam_img)
            plt.imshow(cam_img)
            plt.axis('off')  # 隐藏坐标轴
            plt.show()