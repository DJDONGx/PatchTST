import warnings

import torch
from torch.utils.data import DataLoader, ConcatDataset


class DataLoaders:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int=0,
        collate_fn=None,
        shuffle_train = True,
        shuffle_val = False,
        use_multi_dset = False,
    ):
        super().__init__()

        self.use_multi_dset = use_multi_dset
        if self.use_multi_dset:
            self.datasetCls_list = datasetCls
            self.dataset_kwargs_list = dataset_kwargs
            for dataset_kwargs in self.dataset_kwargs_list:
                if "split" in dataset_kwargs.keys():
                    del dataset_kwargs["split"]
        else:
            self.datasetCls = datasetCls
            if "split" in dataset_kwargs.keys():
                del dataset_kwargs["split"]
            self.dataset_kwargs = dataset_kwargs
        self.batch_size = batch_size
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val


        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()        
 
        
    def train_dataloader(self):
        if self.use_multi_dset:
            return self._make_multi_dloader("train", shuffle=self.shuffle_train)
        else:
            return self._make_dloader("train", shuffle=self.shuffle_train)

    def val_dataloader(self):
        if self.use_multi_dset:
            return self._make_multi_dloader("val", shuffle=self.shuffle_val)
        else:
            return self._make_dloader("val", shuffle=self.shuffle_val)

    def test_dataloader(self):
        if self.use_multi_dset:
            return self._make_multi_dloader("test", shuffle=False)
        else:
            return self._make_dloader("test", shuffle=False)

    def _make_dloader(self, split, shuffle=False):
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)
        if len(dataset) == 0: return None
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )


    def _make_multi_dloader(self, split, shuffle=False):
        dataset_list = []
        for i in range(len(self.datasetCls_list)):
            dataset = self.datasetCls_list[i](**self.dataset_kwargs_list[i], split=split)
            dataset_list.append(dataset)
        # print("每个数据集的patch数量")
        # for i in dataset_list:
        #     print(len(i))
        concat_dataset = ConcatDataset(dataset_list)
        # print("合并数据集的patch数量")
        # print(len(concat_dataset))

        # a = dataset_list[1][len(dataset_list[1]) - 1]
        # b = concat_dataset[-1]
        # print((a[0] == b[0]).all())
        # exit()
        return DataLoader(
            concat_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )


    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )

    def add_dl(self, test_data, batch_size=None, **kwargs):
        # check of test_data is already a DataLoader
        from ray.train.torch import _WrappedDataLoader
        if isinstance(test_data, DataLoader) or isinstance(test_data, _WrappedDataLoader): 
            return test_data

        # get batch_size if not defined
        if batch_size is None: batch_size=self.batch_size        
        # check if test_data is Dataset, if not, wrap Dataset
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)        
        
        # create a new DataLoader from Dataset 
        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data

    
