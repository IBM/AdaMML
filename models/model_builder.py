from . import (resnet, sound_mobilenet_v2,
               joint_resnet_mobilenetv2, adamml)

MODEL_TABLE = {
    'resnet': resnet,
    'sound_mobilenet_v2': sound_mobilenet_v2,
    'joint_resnet_mobilenetv2': joint_resnet_mobilenetv2,
    'adamml': adamml
}


def build_model(args, test_mode=False):
    """
    Args:
        args: all options defined in opts.py and num_classes
        test_mode:
    Returns:
        network model
        architecture name
    """
    model = MODEL_TABLE[args.backbone_net](**vars(args))
    network_name = model.network_name if hasattr(model, 'network_name') else args.backbone_net

    if isinstance(args.modality, list):
        modality = '-'.join([x for x in args.modality])
    else:
        modality = args.modality

    arch_name = "{dataset}-{modality}-{arch_name}".format(
        dataset=args.dataset, modality=modality, arch_name=network_name)
    arch_name += "-f{}".format(args.groups)
    if args.dense_sampling:
        arch_name += "-s{}".format(args.frames_per_group)


    # add setting info only in training
    if not test_mode:
        arch_name += "-{}{}-bs{}{}-e{}".format(args.lr_scheduler, "-syncbn" if args.sync_bn else "",
                                             args.batch_size, '-' + args.prefix if args.prefix else "", args.epochs)
    return model, arch_name
