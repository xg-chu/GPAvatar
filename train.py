#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (purkialo@gmail.com)

import os
import sys
import argparse
import warnings
sys.path.append('./')

from core.tools.config import ConfigDict
from core.utils.utils import device_parser
from core.tools.lightning_engine import LightningEngine

def main(args):
    # build config
    config_dict = ConfigDict(
        model_config_path=os.path.join('./configs/model', args.config+'.yaml'), 
        data_config_path=os.path.join('./configs/data', args.dataset+'.yaml')
    )
    config_dict['DEVICES'] = device_parser(args.devices)
    # config_dict.merge_command_line(vars(args))
    lightning = LightningEngine(config_dict, args.debug)
    # lightning = OptiCamEngine(config_dict, args.debug)
    lightning.run(args.basemodel)


if __name__ == "__main__":
    from tqdm.std import TqdmExperimentalWarning
    from pytorch_lightning.utilities.warnings import PossibleUserWarning
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=PossibleUserWarning, lineno=0, append=False)
    # build args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--devices', '-d', default='cpu', type=str)
    parser.add_argument('--basemodel', default=None, type=str)
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))
    # launch
    main(args)
