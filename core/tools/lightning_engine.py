#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
import pytorch_lightning as ptl
import core.utils.distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from core.tools.builder import build_model, build_dataset

class LightningEngine:
    def __init__(self, config_dict, debug=False):
        torch.set_float32_matmul_precision('high')
        # build dataset
        self.train_dataloader, meta_info = build_dataset(
            config_dict['TRAIN']['GENERAL']['BATCH_SIZE_PER_GPU'], 
            config_dict['DATASET'], config_dict['DATASET_AUGMENT'], 
            stage='train'
        )
        self.val_dataloader, _ = build_dataset(
            1, config_dict['DATASET'], config_dict['DATASET_AUGMENT'], 
            stage='val', num_workers=1, slice= 8 if debug else None
        )
        # build model
        model = build_model(config_dict, meta_info)
        self.model = model
        # loop config
        length = min(config_dict['TRAIN']['GENERAL']['MODEL_DUMP_INTERVAL'], len(self.train_dataloader)) if not debug else 30
        total_epoch = config_dict['TRAIN']['GENERAL']['TRAIN_ITER'] // length + 1
        loop_config = dict(
            max_epochs=total_epoch, 
            check_val_every_n_epoch=1, num_sanity_val_steps=0,
            limit_train_batches=length
        )
        # logger config
        logger = ptl.loggers.TensorBoardLogger(
            save_dir="outputs", name=config_dict['TRAIN']['GENERAL']['EXP_STR'],
            version=config_dict['TRAIN']['GENERAL']['TIME_STR']
        )
        logger_config = dict(
            log_every_n_steps=50, callbacks=[RichProgressBar()], 
            logger=False if debug else logger, enable_checkpointing=False,
        )
        self.lightning_trainer = ptl.Trainer(
            devices=config_dict['DEVICES'], accelerator="gpu", # strategy="auto",
            strategy="ddp_find_unused_parameters_true", # precision="16-mixed",
            **loop_config, **logger_config, 
        )
        self.config_dict = config_dict

    def run(self, basemodel):
        if self.lightning_trainer.is_global_zero:
            print(self.config_dict)
        if basemodel is not None:
            ckpt = torch.load(basemodel, map_location='cpu')
            self.model.load_state_dict(ckpt['state_dict'], strict=False)
        self.lightning_trainer.fit(
            model=self.model, 
            train_dataloaders=self.train_dataloader, 
            val_dataloaders=self.val_dataloader,
        )
