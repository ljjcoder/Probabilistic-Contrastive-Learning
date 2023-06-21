from clsda.loader.cls_loaders import process_one_cls_dataset


def parse_args_for_dataset(dataset_args, debug=False, train_debug_sample_num=10,
                           test_debug_sample_num=10, random_seed=1234, data_root=None, task_type=None):
    """

    :param dataset_args:
    :param debug:
    :param train_debug_sample_num:
    :param test_debug_sample_num:
    :return: 返回一个list
    """
    if debug:
        print("YOU ARE IN DEBUG MODE!!!!!!!!!!!!!!!!!!!")

    # Setup task type
    if task_type == 'cls':
        process_one_dataset = process_one_cls_dataset
    else:
        raise RuntimeError('wrong dataset task name {}'.format(task_type))

    # Setup Augmentations
    trainset_args = dataset_args['train']
    testset_args = dataset_args['test']
    train_augmentations = trainset_args.get('pipeline', None)
    test_augmentations = testset_args.get('pipeline', None)
    # Setup Dataloader
    # 其它参数
    train_batchsize = trainset_args.get('batch_size',None)
    test_batchsize = testset_args.get('batch_size',None)
    n_workers = dataset_args['n_workers']
    drop_last = dataset_args.get('drop_last', True)

    # 训练集
    train_loaders = []
    for i in range(1, 100):
        if i in trainset_args.keys():
            temp_train_aug = trainset_args[i].get('augmentation', None)
            temp_train_aug = train_augmentations if temp_train_aug is None else temp_train_aug
            temp_train_loader = process_one_dataset(trainset_args[i], pipelines=temp_train_aug,
                                                    batch_size=train_batchsize, n_workers=n_workers, shuffle=True,
                                                    debug=debug,
                                                    sample_num=train_debug_sample_num, drop_last=drop_last,
                                                    data_root=data_root,
                                                    random_seed=random_seed)
            train_loaders.append(temp_train_loader)
        else:
            break

    # 测试集
    test_loaders = []
    for i in range(1, 100):
        if i in testset_args.keys():
            temp_test_aug = testset_args[i].get('augmentation', None)
            temp_test_aug = test_augmentations if temp_test_aug is None else temp_test_aug
            temp_test_loader = process_one_dataset(testset_args[i], pipelines=temp_test_aug,
                                                   batch_size=test_batchsize,
                                                   n_workers=n_workers,
                                                   shuffle=False, debug=debug,
                                                   sample_num=test_debug_sample_num,
                                                   drop_last=False, data_root=data_root,
                                                   random_seed=random_seed,
                                                   )
            test_loaders.append(temp_test_loader)

    return train_loaders, test_loaders
