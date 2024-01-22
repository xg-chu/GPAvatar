# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

from fvcore.common.registry import Registry  # for backward compatibility.

'''
    BACKBONE_REGISTRY = Registry('BACKBONE')
    Usage:
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
                ...
    Or:
        BACKBONE_REGISTRY.register(MyBackbone)
'''

MODEL_REGISTRY = Registry("MODEL")
DATA_BUILDER_REGISTRY = Registry("DATA_BUILDER")
