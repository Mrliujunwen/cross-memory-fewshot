import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.dataloader.samplers import CategoriesSampler
from Models.utils import *
from Models.dataloader.data_utils import *
from Models.models.Network import CSM
import matplotlib.pyplot as plt

from torch.autograd import Variable
import utils
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time
import os
import numpy as np

from thop import profile

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'cub','tieredimagenet','fc100','tieredimagenet','cifar_fs'])
    parser.add_argument('-data_dir', type=str, default=DATA_DIR,help='dir of datasets')
    parser.add_argument('-set',type=str,default='test',choices=['test','val'],help='the set used for validation')
    parser.add_argument('-bs', type=int, default=1,help='batch size of tasks')
    parser.add_argument('-max_epoch', type=int, default=200)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-temperature', type=float, default=12.5)
    parser.add_argument('-step_size', type=int, default=20)
    parser.add_argument('-gamma', type=float, default=0.5)
    parser.add_argument('-val_frequency',type=int,default=50)
    parser.add_argument('-random_val_task',action='store_true',help='random samples tasks for validation at each epoch')
    parser.add_argument('-save_all',default=True,action='store_true',help='save models on each epoch')
    parser.add_argument('-way', type=int, default=5)
    parser.add_argument('-shot', type=int, default=1)
    parser.add_argument('-query', type=int, default=15)
    parser.add_argument('-val_episode', type=int, default=600)
    parser.add_argument('-test_episode', type=int, default=2000)
    parser.add_argument('-pretrain_dir', type=str, default=PRETRAIN_DIR)
    parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
    parser.add_argument('-norm', type=str, default='center', choices=['center'], help='feature normalization')
    parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
    parser.add_argument('-feature_pyramid', type=str, default=None, help='you can set it like: 2,3')
    parser.add_argument('-num_patch',type=int,default=9)
    parser.add_argument('-patch_list',type=str,default='2,3',help='the size of grids at every image-pyramid level')
    parser.add_argument('-patch_ratio',type=float,default=2,help='scale the patch to incorporate context around the patch')
    parser.add_argument('-solver', type=str, default='opencv', choices=['opencv', 'qpth'])
    parser.add_argument('-form', type=str, default='L2', choices=['QP', 'L2'])
    parser.add_argument('-l2_strength', type=float, default=0.000001)
    parser.add_argument('-sfc_lr', type=float, default=0.1, help='learning rate of SFC')
    parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
    parser.add_argument('-sfc_update_step', type=float, default=100, help='number of updating step of SFC')
    parser.add_argument('-sfc_bs', type=int, default=4, help='batch size for finetune sfc')
    parser.add_argument('-temperature2', type=float, default=1.0)
    parser.add_argument('-alpha', type=float, default=0.7, help='the balanced parameters between loss function')
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-extra_dir', type=str,default=None,help='extra information that is added to checkpoint dir')
    parser.add_argument('-seed', type=int, default=1)
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
    parser.add_argument('--model_type', type=str, default='resnet12')
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--csm_mode', type=str, default='fcn')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dense', type=bool, default=False)
    parser.add_argument('--num_points', type=int, default=100)
    parser.add_argument('--grid_size', type=int, default=5)
    return parser.parse_args()

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_path = args.save_path if args.save_path is not None else './experiments/meta_train'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    set_seed(args.seed)
    num_gpu = set_gpu(args)
    Dataset=set_up_datasets(args)

    args.pretrain_dir=osp.join(args.pretrain_dir,'%s/resnet12/max_acc.pth'%(args.dataset))
    model = CSM(args)
    if args.init_weights is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.init_weights)['params']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = nn.DataParallel(model, list(range(num_gpu)))
    model = model.cuda()
    model.train()

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, args.val_frequency*args.bs, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)

    valset = Dataset(args.set, args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=0, pin_memory=True)

    if not args.random_val_task:
        print ('fix val set for all epochs')
    print('save all checkpoint models:', (args.save_all is True))

    label = torch.arange(args.way, dtype=torch.int8).repeat(args.query)
    label = label.type(torch.LongTensor)
    label = label.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total - ', pytorch_total_params)
    print('Trainable - ', trainable_pytorch_total_params)
    with autocast():
        def save_model(name):
            torch.save(dict(params=model.state_dict()), osp.join(save_path, name + '.pth'))

        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        global_count = 0
        writer = SummaryWriter(osp.join(save_path,'tf'))

        result_list=[save_path]
        for epoch in range(1, args.max_epoch + 1):
            print (save_path)
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
                logits, logits_trans = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))

                loss = args.alpha * F.cross_entropy(logits, label) + (1 - args.alpha) * F.cross_entropy(logits_trans, label)
                acc = count_acc(logits, label)

                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)

                total_loss = loss/args.bs#batch of tasks, done by accumulate gradients
                writer.add_scalar('data/total_loss', float(total_loss), global_count)
                tqdm_gen.set_description('epo {}, total loss={:.4f} acc={:.4f}'
                      .format(epoch, total_loss.item(), acc))
                tl.add(total_loss.item())
                ta.add(acc)
                total_loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            tl = tl.item()
            ta = ta.item()
            vl = Averager()
            va = Averager()

            #validation
            model.eval()
            with torch.no_grad():
                tqdm_gen = tqdm.tqdm(val_loader)
                for i, batch in enumerate(tqdm_gen, 1):
                    data, _ = [_.cuda() for _ in batch]
                    k = args.way * args.shot
                    model.module.mode = 'encoder'
                    data = model(data)
                    data_shot, data_query = data[:k], data[k:]
                    model.module.mode = 'meta'
                    if args.shot > 1:
                        data_shot = model.module.get_sfc(data_shot)
                    logits ,logits_trans = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
                    loss = args.alpha * F.cross_entropy(logits, label) + (1 - args.alpha) * F.cross_entropy(logits_trans, label)
                    acc = count_acc(logits, label)

                    vl.add(loss.item())
                    va.add(acc)

            vl = vl.item()
            va = va.item()
            writer.add_scalar('data/val_loss', float(vl), epoch)
            writer.add_scalar('data/val_acc', float(va), epoch)
            tqdm_gen.set_description('epo {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

            print ('val acc:%.4f'%va)
            if va >= trlog['max_acc']:
                print ('*********A better model is found*********')
                trlog['max_acc'] = va
                trlog['max_acc_epoch'] = epoch
                save_model('max_acc')

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl)
            trlog['val_acc'].append(va)

            result_list.append('epoch:%03d,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f'%(epoch,tl,ta,vl,va))

            torch.save(trlog, osp.join(save_path, 'trlog'))
            if args.save_all:
                save_model('epoch-%d'%epoch)
                torch.save(optimizer.state_dict(), osp.join(save_path,'optimizer_latest.pth'))
            print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            print ('This epoch takes %d seconds'%(time.time()-start_time),'\nstill need %.2f hour to finish'%((time.time()-start_time)*(args.max_epoch-epoch)/3600))
            lr_scheduler.step()

        writer.close()

        # Test Phase
        trlog = torch.load(osp.join(save_path, 'trlog'))
        test_set = Dataset('test', args)
        sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=0, pin_memory=True)
        test_acc_record = np.zeros((args.test_episode,))
        model.load_state_dict(torch.load(osp.join(save_path, 'max_acc' + '.pth'))['params'])
        model.eval()

        ave_acc = Averager()
        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)

        tqdm_gen = tqdm.tqdm(loader)
        with torch.no_grad():
            for i, batch in enumerate(tqdm_gen, 1):
                data, _ = [_.cuda() for _ in batch]
                k = args.way * args.shot
                model.module.mode = 'encoder'
                data = model(data)
                data_shot, data_query = data[:k], data[k:]
                model.module.mode = 'meta'
                if args.shot > 1:
                    data_shot = model.module.get_sfc(data_shot)
                logits, logits_trans = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))

                acc = count_acc(logits, label)* 100

                ave_acc.add(acc)
                test_acc_record[i-1] = acc
                tqdm_gen.set_description('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item(), acc))

        m, pm = compute_confidence_interval(test_acc_record)

        result_list.append('Val Best Epoch {},\nbest val Acc {:.4f}, \nbest est Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
        result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        print (result_list[-2])
        print (result_list[-1])
        save_list_to_txt(os.path.join(save_path,'results.txt'),result_list)

if __name__ == '__main__':
    args = get_args()
    main(args)
