# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import time
import yaml
import random
import colored
from colored import stylize

class ConfigDict(dict):
    def __init__(self, model_config_path=None, data_config_path=None):
        # build new config
        config_dict = read_config(model_config_path)
        if data_config_path is not None:
            dataset_dict = read_config(data_config_path)
            merge_a_into_b(dataset_dict, config_dict)
        # set output path 
        experiment_string = config_dict['MODEL']['MODEL_NAME'] + '_' + \
                            config_dict['DATASET']['DATASET_NAME']
        time_string = time.strftime("%b%d_%H%M_", time.localtime()) + \
                      "".join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
        config_dict['TRAIN']['GENERAL']['EXP_STR'] = experiment_string
        config_dict['TRAIN']['GENERAL']['TIME_STR'] = time_string
        self._merged_key = []
        # self.config_dict = config_dict
        super().__init__(config_dict)

    def __str__(self, ):
        return pretty_dict(self, highlight_keys=self._merged_key)

    def merge_command_line(self, command_line_dict):
        import numpy as np
        for key in list(command_line_dict.keys()):
            if key in ['config', 'dataset', 'debug']:
                continue
            elif key == 'opts':
                opts = np.array(command_line_dict[key]).reshape(-1, 2)
                for k, v in opts:
                    self._merge(self.config_dict, k, v)
            elif command_line_dict[key] is not None:
                self._merged_key.append(key.upper())
                self.config_dict[key.upper()] = command_line_dict[key]

    def _merge(self, config_dict, target_key, target_value):
        if_find = False
        for key, v in config_dict.items():
            if isinstance(v, dict):
                self._merge(config_dict[key], target_key, target_value)
            elif key == target_key.upper():
                self._merged_key.append(key)
                if isinstance(config_dict[key], int):
                    config_dict[key] = int(target_value)
                elif isinstance(config_dict[key], float):
                    config_dict[key] = float(target_value)
                elif isinstance(config_dict[key], bool):
                    config_dict[key] = bool(target_value)
                else:
                    config_dict[key] = target_value
        return if_find


def read_config(path):
    if isinstance(path, dict):
        return path
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} was not found.")
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def merge_a_into_b(a, b):
    # merge dict a into dict b. values in a will overwrite b.
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(
                b[k], dict
            ), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


def pretty_dict(input_dict, indent=0, highlight_keys=[]):
    out_line = ""
    tab = "    "
    for key, value in input_dict.items():
        if key in highlight_keys:
            out_line += tab * indent + stylize(str(key), colored.fg(1))
        else:
            out_line += tab * indent + stylize(str(key), colored.fg(2))
        if isinstance(value, dict):
            out_line += ':\n'
            out_line += pretty_dict(value, indent+1, highlight_keys)
        else:
            if key in highlight_keys:
                out_line += ":" + "\t" + stylize(str(value), colored.fg(1)) + '\n'
            else:
                out_line += ":" + "\t" + stylize(str(value), colored.fg(2)) + '\n'
    if indent == 0:
        max_length = 0
        for line in out_line.split('\n'):
            max_length = max(max_length, len(line.split('\t')[0]))
        max_length += 4
        aligned_line = ""
        for line in out_line.split('\n'):
            if '\t' in line:
                aligned_number = max_length - len(line.split('\t')[0])
                line = line.replace('\t',  aligned_number * ' ')
            aligned_line += line+'\n'
        return aligned_line[:-2]
    return out_line
