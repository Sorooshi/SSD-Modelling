import numpy as np
from typing import Tuple
from typing import Dict
from collections import defaultdict
from pathlib import Path
from ssd_sim.data.ssd_data import SSDDATA
from sklearn.preprocessing import QuantileTransformer


class PreProcess:
    def __init__(self, data, out_dist: str="normal",):
        self.data = data
        self.out_dist = out_dist
        self.data_pp = defaultdict(list)
        self.data_pp_revert = defaultdict(list)

    def pre_process(self, ):
        for k, v in self.data.items():
            v = np.asarray(v)
            x_q, QT = self._quantile_standardizer(x=v)
            self.data_pp[k] = {}
            self.data_pp[k]["arr"] = x_q
            self.data_pp[k]["_qt"] = QT

        return self.data_pp

    def revert_pre_process(self, ):
        for k, v in self.data_pp.items():
            x = v["arr"]
            QT = v["_qt"]
            self.data_pp_revert[k] = self._inverse_quantile_standardizer(QT=QT, x=x)

        return self.data_pp_revert

    def _quantile_standardizer(self, x):
        QT = QuantileTransformer(output_distribution=self.out_dist, )
        x_q = QT.fit_transform(x)
        return x_q, QT

    # if one asks that the "QT obj" for test data
    # should be the same as "QT obj" for train data
    def _fit_quantile_standardizer(self, QT, x):
        return QT.fit_transform(x)

    def _inverse_quantile_standardizer(self, QT, x):
        return QT.inverse_transform(x)

