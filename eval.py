import argparse

import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from Models.dataloader.samplers import CategoriesSampler
# from Models.models.yuanNetwork import DeepEMD
from Models.utils import *
from Models.dataloader.data_utils import *
PRETRAIN_DIR='deepemd_pretrain_model/'
# DATA_DIR='/home/zhangchi/dataset'
DATA_DIR=r'F:\few-shot dataset'
from Models.models.attNetwork import DeepEMD
# from Models.models.renet import DeepEMD

# DATA_DIR='your/default/dataset/dir'
# # DATA_DIR='/home/zhangchi/dataset'
# MODEL_DIR='deepemd_trained_model/miniimagenet/fcn/max_acc.pth'
#
from torch.cuda.amp import autocast as autocast


parser = argparse.ArgumentParser()
# about task
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=15)  # number of query image per class
parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'cub','tieredimagenet','fc100','tieredimagenet_yao','cifar_fs'])
parser.add_argument('-set', type=str, default='test', choices=['train','val', 'test'])
parser.add_argument('-data_dir', type=str, default=DATA_DIR,help='dir of datasets')

# about model
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-metric', type=str, default='cosine', choices=[ 'cosine' ])
parser.add_argument('-norm', type=str, default='center', choices=[ 'center'])
parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
#deepemd fcn only
parser.add_argument('-feature_pyramid', type=str, default=None)
#deepemd sampling only
parser.add_argument('-num_patch',type=int,default=9)
#deepemd grid only patch_list
parser.add_argument('-patch_list',type=str,default='2,3')
parser.add_argument('-patch_ratio',type=float,default=2)
# solver
parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
# SFC
parser.add_argument('-sfc_lr', type=float, default=0.1, help='learning rate of SFC')
parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
parser.add_argument('-sfc_update_step', type=float, default=100, help='number of updating step of SFC')
parser.add_argument('-sfc_bs', type=int, default=4, help='batch size for finetune sfc')
parser.add_argument('-temperature2', type=float, default=1.0)
parser.add_argument('-alpha', type=float, default=0.7, help='the balanced parameters between loss function')
parser.add_argument('-model_dir',  default=r'F:\paper\DeepEMD-master\checkpoint\miniimagenet\fcn\1shot-5way\epoch-100.pth')

# OTHERS
parser.add_argument('-gpu', default='0')
parser.add_argument('-extra_dir', type=str,default=None,help='extra information that is added to checkpoint dir, e.g. hyperparameters')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')

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
if args.feature_pyramid is not None:
    args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
args.patch_list = [int(x) for x in args.patch_list.split(',')]

pprint(vars(args))
set_seed(args.seed)
num_gpu = set_gpu(args)
Dataset=set_up_datasets(args)
# with autocast():

    # model
model = DeepEMD(args)
model = load_model(model, args.model_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()

# test dataset
test_set = Dataset(args.set, args)
sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
loader = DataLoader(test_set, batch_sampler=sampler, num_workers=0, pin_memory=True)
tqdm_gen = tqdm.tqdm(loader)

# label of query images
ave_acc = Averager()
test_acc_record = np.zeros((args.test_episode,))
label = torch.arange(args.way).repeat(args.query)
label = label.type(torch.cuda.LongTensor)

with torch.no_grad():
    for i, batch in enumerate(tqdm_gen, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        model.module.mode = 'encoder'
        data = model(data)
        data_shot, data_query = data[:k], data[k:]  # shot: 5,3,84,84  query:75,3,84,84
        model.module.mode = 'meta'
        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)
        # logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        logits, logits_trans = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))

        acc = count_acc(logits, label) * 100
        ave_acc.add(acc)
        test_acc_record[i - 1] = acc
        m, pm = compute_confidence_interval(test_acc_record[:i])
        tqdm_gen.set_description('batch {}: This episode:{:.2f}  average: {:.4f}+{:.4f}'.format(i, acc, m, pm))

    m, pm = compute_confidence_interval(test_acc_record)
    result_list = ['test Acc {:.4f}'.format(ave_acc.item())]
    result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))
    print(result_list[0])
    print(result_list[1])
