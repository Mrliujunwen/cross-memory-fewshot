def set_up_datasets(args):
    if args.dataset == 'miniimagenet':
        args.num_class = 64
        if args.csm_mode == 'fcn':
            from Models.dataloader.miniimagenet.fcn.mini_imagenet import MiniImageNet as Dataset
        elif args.csm_mode == 'sampling':
            from Models.dataloader.miniimagenet.sampling.mini_imagenet import MiniImageNet as Dataset
        elif args.csm_mode == 'grid':
            from Models.dataloader.miniimagenet.grid.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'cub':
        args.num_class = 100
        if args.csm_mode == 'fcn':
            from Models.dataloader.cub.fcn.cub import CUB as Dataset
        elif args.csm_mode == 'sampling':
            from Models.dataloader.cub.sampling.cub import CUB as Dataset
        elif args.csm_mode == 'grid':
            from Models.dataloader.cub.grid.cub import CUB as Dataset
    elif args.dataset == 'fc100':
        args.num_class = 60
        if args.csm_mode == 'fcn':
            from Models.dataloader.fc100.fcn.fc100 import DatasetLoader as Dataset
        elif args.csm_mode == 'sampling':
            from Models.dataloader.fc100.sampling.fc100 import DatasetLoader as Dataset
        elif args.csm_mode == 'grid':
            from Models.dataloader.fc100.grid.fc100 import DatasetLoader as Dataset
    elif args.dataset == 'tieredimagenet':
        args.num_class = 351
        if args.csm_mode == 'fcn':
            from Models.dataloader.tieredimagenet.fcn.tiered_imagenet import tieredImageNet as Dataset
        elif args.csm_mode == 'sampling':
            from Models.dataloader.tieredimagenet.sampling.tiered_imagenet import tieredImageNet as Dataset
        elif args.csm_mode == 'grid':
            from Models.dataloader.tieredimagenet.grid.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'cifar_fs':
        args.num_class = 64
        if args.csm_mode == 'fcn':
            from Models.dataloader.cifar_fs.fcn.cifar_fs import DatasetLoader as Dataset
        elif args.csm_mode == 'sampling':
            from Models.dataloader.cifar_fs.sampling.cifar_fs import DatasetLoader as Dataset
        elif args.csm_mode == 'grid':
            from Models.dataloader.cifar_fs.gird.cifar_fs import DatasetLoader as Dataset
    else:
        raise ValueError('Unknown Dataset')
    return Dataset

def get_train_loader(args):
    if args.csm_mode == 'fcn':
        trainset = Dataset('train', args)
    elif args.csm_mode == 'sampling':
        trainset = Dataset_sampling('train', args)
    elif args.csm_mode == 'grid':
        trainset = Dataset_grid('train', args)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return train_loader

def get_val_loader(args):
    if args.csm_mode == 'fcn':
        valset = Dataset('val', args)
    elif args.csm_mode == 'sampling':
        valset = Dataset_sampling('val', args)
    elif args.csm_mode == 'grid':
        valset = Dataset_grid('val', args)
    val_loader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return val_loader

def get_meta_train_loader(args):
    if args.csm_mode == 'fcn':
        train_set = Dataset('train', args)
    elif args.csm_mode == 'sampling':
        train_set = Dataset_sampling('train', args)
    elif args.csm_mode == 'grid':
        train_set = Dataset_grid('train', args)
    sampler = CategoriesSampler(train_set.label, args.train_episode, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=train_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    return train_loader

def get_meta_val_loader(args):
    if args.csm_mode == 'fcn':
        val_set = Dataset('val', args)
    elif args.csm_mode == 'sampling':
        val_set = Dataset_sampling('val', args)
    elif args.csm_mode == 'grid':
        val_set = Dataset_grid('val', args)
    sampler = CategoriesSampler(val_set.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=val_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    return val_loader

def get_meta_test_loader(args):
    if args.csm_mode == 'fcn':
        test_set = Dataset('test', args)
    elif args.csm_mode == 'sampling':
        test_set = Dataset_sampling('test', args)
    elif args.csm_mode == 'grid':
        test_set = Dataset_grid('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(dataset=test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    return test_loader
