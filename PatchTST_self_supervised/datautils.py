

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange'
        ]

def get_dls(params):
    # 多数据集
    if params.multi_dset:
        print("-" * 25, "处理多数据集", params.multi_dset, "-" * 25)
        return get_multi_dls(params)

    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    if params.dset == 'ettm1':
        # root_path = '/data/datasets/public/ETDataset/ETT-small/'
        root_path = './dataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'ettm2':
        # root_path = '/data/datasets/public/ETDataset/ETT-small/'
        root_path = './dataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'etth1':
        # root_path = '/data/datasets/public/ETDataset/ETT-small/'
        root_path = './dataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'etth2':
        # root_path = '/data/datasets/public/ETDataset/ETT-small/'
        root_path = './dataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    

    elif params.dset == 'electricity':
        # root_path = '/data/datasets/public/electricity/'
        root_path = './dataset/electricity/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'traffic':
        # root_path = '/data/datasets/public/traffic/'
        root_path = './dataset/traffic/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    
    elif params.dset == 'weather':
        # root_path = '/data/datasets/public/weather/'
        root_path = './dataset/weather/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'illness':
        # root_path = '/data/datasets/public/illness/'
        root_path = './dataset/illness/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'exchange':
        # root_path = '/data/datasets/public/exchange_rate/'
        root_path = './dataset/exchange_rate/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls


def get_multi_dls(params):
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    datasetCLS_list = []
    dataset_kwargs_list = []
    for dataset_name in params.multi_dset:
        datasetCLS, dataset_kwargs = get_datasetCls_and_kwargs(dataset_name, params)
        datasetCLS_list.append(datasetCLS)
        dataset_kwargs_list.append(dataset_kwargs)
    dls = DataLoaders(
        datasetCls=datasetCLS_list,
        dataset_kwargs=dataset_kwargs_list,
        batch_size=params.batch_size,
        workers=params.num_workers,
        use_multi_dset=True,
    )
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls

def get_datasetCls_and_kwargs(dataset_name, params):
    if dataset_name == 'ettm1':
        root_path = './dataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dataset_kwargs = {
            'root_path': root_path,
            'data_path': 'ETTm1.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        }
        return Dataset_ETT_minute, dataset_kwargs

    elif dataset_name == 'ettm2':
        root_path = './dataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dataset_kwargs = {
            'root_path': root_path,
            'data_path': 'ETTm2.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        }
        return Dataset_ETT_minute, dataset_kwargs

    elif dataset_name == 'etth1':
        root_path = './dataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dataset_kwargs = {
            'root_path': root_path,
            'data_path': 'ETTh1.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        }
        return Dataset_ETT_hour, dataset_kwargs

    elif dataset_name == 'etth2':
        root_path = './dataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dataset_kwargs = {
            'root_path': root_path,
            'data_path': 'ETTh2.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        }
        return Dataset_ETT_hour, dataset_kwargs

    elif dataset_name == 'electricity':
        root_path = './dataset/electricity/'
        size = [params.context_points, 0, params.target_points]
        dataset_kwargs = {
            'root_path': root_path,
            'data_path': 'electricity.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        }
        return Dataset_Custom, dataset_kwargs

    elif dataset_name == 'traffic':
        root_path = './dataset/traffic/'
        size = [params.context_points, 0, params.target_points]
        dataset_kwargs = {
            'root_path': root_path,
            'data_path': 'traffic.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        }
        return Dataset_Custom, dataset_kwargs

    elif dataset_name == 'weather':
        root_path = './dataset/weather/'
        size = [params.context_points, 0, params.target_points]
        dataset_kwargs = {
            'root_path': root_path,
            'data_path': 'weather.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        }
        return Dataset_Custom, dataset_kwargs

    elif dataset_name == 'illness':
        root_path = './dataset/illness/'
        size = [params.context_points, 0, params.target_points]
        dataset_kwargs = {
            'root_path': root_path,
            'data_path': 'national_illness.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        }
        return Dataset_Custom, dataset_kwargs

    elif dataset_name == 'exchange':
        root_path = './dataset/exchange_rate/'
        size = [params.context_points, 0, params.target_points]
        dataset_kwargs = {
            'root_path': root_path,
            'data_path': 'exchange_rate.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        }
        return Dataset_Custom, dataset_kwargs


if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
