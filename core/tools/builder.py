# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch

from core.utils.registry import MODEL_REGISTRY, DATA_BUILDER_REGISTRY

def build_model(config_dict, data_meta_info):
    model_name = config_dict['MODEL']['MODEL_NAME']
    model = MODEL_REGISTRY.get(model_name)(config_dict, data_meta_info)
    return model


def build_dataset(
        batch_size_per_gpu, dataset_config, augment_config, 
        stage, num_workers=4, shuffle=None, slice=None, auto_batch=False
    ):
    dataset_name = dataset_config['DATASET_LOADER']
    dataset = DATA_BUILDER_REGISTRY.get(dataset_name)(dataset_config, augment_config, stage)
    if slice is not None:
        dataset.slice(slice)
    if auto_batch:
        for i in [1, 2, 4, 8, 16]:
            if len(dataset) % i == 0:
                batch_size_per_gpu = i
            else:
                break
        print('AUTO BATCH SIZE: {}.'.format(batch_size_per_gpu))
    
    if_shuffle = shuffle if shuffle is not None else stage=='train'
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_per_gpu,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None, shuffle=if_shuffle, 
    )
    return dataloader, dataset.meta_info


def get_meta_info(dataset_config):
    dataset_name = dataset_config['DATASET_LOADER']
    dataset = DATA_BUILDER_REGISTRY.get(dataset_name)(dataset_config, None, stage='train')
    return dataset.meta_info
