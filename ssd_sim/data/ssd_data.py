import re
import logging
from pathlib import Path
from itertools import chain
from typing import List, Tuple
from collections import defaultdict


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class SSDDATA:

    def __init__(self, paths: Tuple[Path], use_last_n_test_split: int=50):
        self.path = paths
        self.visualization_data = list()
        self.logger = logging.getLogger(__name__)
        self.files_ = defaultdict(list)
        self.data_grouped = defaultdict(list)
        self.data_grouped_kf = defaultdict(list)
        self.use_last_n_test_split = use_last_n_test_split
        self.train_test_data = defaultdict(list)

        files = {}
        for i, item in enumerate(paths):
            item = Path(item)
            files[i] = list(item.iterdir())
        self.all_data = self._make_df(list(chain(*files.values())))  # all data in one df

        # To get a split with n_last_data_points reserved for test/visualization
        _tmp_train = []
        _tmp_test = []
        for i, v in enumerate(files.values()):
            _tmp_train.append(v[:-self.use_last_n_test_split])
            _tmp_test.append(v[-self.use_last_n_test_split:])
        self.files_["train"] = list(chain(*_tmp_train))
        self.files_["test"] = list(chain(*_tmp_test))

    # not grouped, not split
    def get_all_data(self, ):
        return self.all_data

    # not grouped data but split into train and test
    def get_train_test_data(self, ):
        self.train_test_data["train_data"] = self._make_df(self.files_["train"])
        self.train_test_data["test_data"] = self._make_df(self.files_["test"])
        return self.train_test_data

    def get_visualization_data(self, ):
        self.visualization_data = self._make_lst(self.files_["test"])
        return self.visualization_data

    # grouped (seq, rnd, iops, lat) and split data
    def get_grouped_data(self, data):

        x_cols = ['iodepth', 'block_size', 'read_fraction', 'io_type', 'load_type']
        # random data
        rnd = data.loc[data["load_type"] == 0.]
        rnd_x = rnd.loc[:, x_cols]
        rnd_iops = pd.DataFrame(rnd.loc[:, "iops"])
        rnd_lat = pd.DataFrame(rnd.loc[:, "latency"])
        self.data_grouped["rnd_x"] = rnd_x
        self.data_grouped["rnd_iops"] = rnd_iops
        self.data_grouped["rnd_lat"] = rnd_lat

        # sequential data
        seq = data.loc[data["load_type"] == 1.]
        seq_x = seq.loc[:, x_cols]
        seq_iops = pd.DataFrame(seq.loc[:, "iops"])
        seq_lat = pd.DataFrame(seq.loc[:, "latency"])

        self.data_grouped["seq_x"] = seq_x
        self.data_grouped["seq_iops"] = seq_iops
        self.data_grouped["seq_lat"] = seq_lat

        return self.data_grouped

    # grouped (seq, rnd, iops, lat) and split data with crosss_validation folds
    def get_grouped_data_folds(self, n_folds, ):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=123)
        for i in ["rnd", "seq"]:
            k = 1
            for train_index, test_index in kf.split(self.data_grouped[i+"_x"]):

                self.data_grouped_kf[i+"_x_train_"+str(k)] = self.data_grouped[i+"_x"].iloc[train_index, :]
                self.data_grouped_kf[i+"_x_test_"+str(k)] = self.data_grouped[i+"_x"].iloc[test_index, :]

                self.data_grouped_kf[i+"_iops_train_"+str(k)] = self.data_grouped[i+"_iops"].iloc[train_index]
                self.data_grouped_kf[i+"_iops_test_"+str(k)] = self.data_grouped[i+"_iops"].iloc[test_index]

                self.data_grouped_kf[i+"_lat_train_"+str(k)] = self.data_grouped[i+"_lat"].iloc[train_index]
                self.data_grouped_kf[i+"_lat_test_"+str(k)] = self.data_grouped[i+"_lat"].iloc[test_index]
                k += 1

        return self.data_grouped_kf

    def _make_df(self, files: List[Path],):
        data = []
        for aname in files:
            try:
                load_type = re.findall(r"__load_type=...", str(aname))[0].split("=")[-1]
                adata = self._read_single_run(str(aname))
                adata["run"] = "_".join(
                    [aname.name.split("__")[0].split("_")[-1], load_type]
                )
                data.append(adata)
            except Exception as e:
                self.logger.info(f"Can not read data in: {aname}")
                self.logger.info(e)
        return pd.concat(data)

    def _make_lst(self, files: List[Path],):
        data = []
        for aname in files:
            try:
                load_type = re.findall(r"__load_type=...", str(aname))[0].split("=")[-1]
                adata = self._read_single_run(str(aname))
                adata["run"] = "_".join(
                    [aname.name.split("__")[0].split("_")[-1], load_type]
                )
                data.append(adata)
            except Exception as e:
                self.logger.info(f"Can not read data in: {aname}")
                self.logger.info(e)
        return data

    def _read_single_run(self, data_dir, ):

        data = []
        iodepth = float(re.findall("iodepth=\d+\.*\d*", data_dir)[0][8:])
        block_size = float(re.findall("block_size=\d+\.*\d*", data_dir)[0][11:])
        # offset = np.float(re.findall("offset=\d+\.*\d*", data_dir)[0][7:])
        read_fraction = float(re.findall("read_fraction=\d+\.\d*", data_dir)[0][14:])

        if '=seq_' in data_dir:
            pre = 'seq'
            load_type = 1
        elif '=rnd_' in data_dir:
            load_type = 0
            pre = 'rnd'

        # IOPS
        iops = []
        for i in range(1, 9):
            iops.append(pd.read_csv(data_dir + f'/result/{pre}_rw_iops.{i}.log',
                                    header=None, usecols=[1, 2]))

        iops = pd.concat(iops, axis=1)

        _iops_val = np.asarray(iops.iloc[:, 0::2].sum(axis=1)).reshape(-1, 1)
        _iops_type = np.asarray(iops.iloc[:, 1::2].mean(axis=1)).reshape(-1, 1)

        iops_np = np.concatenate((_iops_val, _iops_type), axis=1)

        iops = pd.DataFrame(data=iops_np, columns=['iops', 'type'])

        lat = []
        for i in range(1, 9):
            lat.append(pd.read_csv(data_dir + f'/result/{pre}_rw_lat.{i}.log',
                                   header=None, usecols=[1, 2]))
        lat = pd.concat(lat)

        # Latency
        _lat_val = np.asarray(lat.iloc[:, 0::2].mean(axis=1)).reshape(-1, 1)
        _lat_type = np.asarray(lat.iloc[:, 1::2].mean(axis=1)).reshape(-1, 1)

        lat_np = np.concatenate((_lat_val, _lat_type), axis=1)

        lat = pd.DataFrame(data=lat_np, columns=['lat', 'type'])

        load_type = re.findall(r'__load_type=...', data_dir)[0].split('=')[-1]

        for io_type in [0, 1]:
            iops_type = iops[iops['type'] == io_type]['iops'].values
            lat_type = lat[lat['type'] == io_type]['lat'].values

            n_obs = min(len(iops_type), len(lat_type))
            iops_type = iops_type[:n_obs]
            lat_type = lat_type[:n_obs]

            data_io = pd.DataFrame()
            data_io['iodepth'] = [iodepth] * n_obs
            data_io['block_size'] = [block_size] * n_obs
            data_io['read_fraction'] = [read_fraction] * n_obs
            # data_io['offset'] = [offset] * n_obs
            data_io['io_type'] = [io_type] * n_obs
            # load_type := 1 >> seq; load_type := 2 >> rnd; load_type := 3 >> o.w
            if load_type.lower() == "seq":
                lt = 1
            elif load_type.lower() == "rnd":
                lt = 0
            else:
                lt = 2
                print("wrong load type!")
            data_io['load_type'] = [lt] * n_obs  # 0 for random, 1 for sequential

            data_io['iops'] = iops_type
            data_io['latency'] = lat_type

            data.append(data_io)
        data = pd.concat(data, axis=0)

        return data
