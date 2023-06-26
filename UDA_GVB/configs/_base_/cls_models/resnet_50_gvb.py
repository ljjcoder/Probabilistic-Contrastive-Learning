backbone_optimizer = dict(
    type='SGD',
    lr=0.001,
    weight_decay=0.001,
    momentum=0.9,
    nesterov=True,
)

backbone = dict(
    type='GVBResNetFc',
    resnet_name='ResNet50',
    class_num=65,
    optimizer=backbone_optimizer,
)

discriminator_optimizer = dict(
    type='SGD',
    lr=0.001,
    weight_decay=0.001,
    momentum=0.9,
    nesterov=True,
)

discriminator = dict(
    type='GVBAdversarialNetwork',
    in_feature=65,
    hidden_size=1024,
    optimizer=discriminator_optimizer,
)

scheduler = dict(
    type='InvLR',
    gamma=0.0001,
    power=0.75,
)

models = dict(
    base_model=backbone,
    discriminator=discriminator,
    lr_scheduler=scheduler,
)
