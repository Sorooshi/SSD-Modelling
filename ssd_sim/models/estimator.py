import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from ssd_sim.common import utils
from types import SimpleNamespace
from ssd_sim.common import metrics
from collections import defaultdict
from ssd_sim.data.ssd_data import SSDDATA
from ssd_sim.data.preprocess import PreProcess

# from ssd_sim.models.flow import NFFitter
# from ssd_sim.common.utils import test_pipeline


class XGBoostReg:  # << NFReg

    def __init__(self, configs: Dict):
        self.visualization_data = list()
        self.models = defaultdict(list)
        self.results = defaultdict(list)
        self.predictions = defaultdict(list)
        self.grouped_data = defaultdict(list)
        self.grouped_data_kf = defaultdict(list)
        self.grouped_data_kf_pp = defaultdict(list)
        self.grouped_train_test_pp = defaultdict(list)

        self.configs = SimpleNamespace(**configs)
        self.logging = "Silent" if not self.configs.logging else "Info"

        self.data = SSDDATA(
            self.configs.paths,
            use_last_n_test_split=self.configs.use_last_n_test_split,
        )

        self.save_path = Path(self.configs.save_path)
        self.report_path = Path(self.configs.report_path)

        if not self.save_path.exists():
            self.save_path.mkdir()

        if not self.report_path.exists():
            self.report_path.mkdir()

    def get_data(self):

        # K-Fold Train and test splits
        all_data = self.data.get_all_data()
        self.grouped_data = self.data.get_grouped_data(data=all_data)
        self.grouped_data_kf = self.data.get_grouped_data_folds(
            n_folds=self.configs.n_folds,
        )
        pp_kf = PreProcess(
            data=self.grouped_data_kf,
            out_dist="uniform",
        )
        self.grouped_data_kf_pp = pp_kf.pre_process()
        grouped_data_kf_pp_revert = pp_kf.revert_pre_process()  # future usage
        print("grouped_data_kf_pp generated.",)

        train_test_data = self.data.get_train_test_data()
        train_data_grouped = self.data.get_grouped_data(
            data=train_test_data["train_data"]
        )
        pp_train = PreProcess(
            data=train_data_grouped,
            out_dist="uniform",
        )
        self.grouped_train_test_pp["train_data"] = pp_train.pre_process()
        # Test data
        test_data_grouped = self.data.get_grouped_data(
            data=train_test_data["test_data"]
        )
        pp_test = PreProcess(
            data=test_data_grouped,
            out_dist="uniform",
        )
        self.grouped_train_test_pp["test_data"] = pp_test.pre_process()
        print("grouped_train_test_pp generated.",)


        self.visualization_data = self.data.get_visualization_data()
        print("grouped_visualization_pp generated.",)

        if self.configs.cross_validation:
            return self.grouped_data_kf_pp  # , self.visualization_data

        # train and test splits without KFolds
        else:
            return self.grouped_train_test_pp  # , self.visualization_data

    def build_kf_models(self, n_folds):

        for k in range(1, n_folds+1):
            for i in ["rnd", "seq"]:
                for j in ["iops", "lat"]:
                    self.models[i+"_"+j+"_" +str(k)]=xgb.XGBRegressor(
                        objective=self.configs.objective,
                        learning_rate=self.configs.learning_rate,
                        max_depth=self.configs.max_depth,
                        n_estimators=self.configs.n_estimators,

                    )
        return self.models

    def fit_models(self, models_kf=None):

        if self.configs.cross_validation:  # <<  and not self.configs.cross_validation >> !!??

            print("Model fitting with Cross Validation (CV).")

            data_to_fit = self.get_data()
            if not models_kf:
                print("Building models for CV.")
                models_kf = self.build_kf_models(n_folds=self.configs.n_folds)

            for k, v in tqdm(models_kf.items()):
                n_kf = str(k.split("_")[-1])  # number of fold

                if "rnd_iops" in k:
                    print("Modeling rnd_iops",)
                    v.fit(data_to_fit["rnd_x_train_" + n_kf]["arr"],
                          data_to_fit["rnd_iops_train_" + n_kf]["arr"])

                elif "rnd_lat" in k:
                    print("Modeling rnd_lat")
                    v.fit(data_to_fit["rnd_x_train_" + n_kf]["arr"],
                          data_to_fit["rnd_lat_train_" + n_kf]["arr"])

                elif "seq_iops" in k:
                    print("Modeling seq_iops")
                    v.fit(data_to_fit["seq_x_train_" + n_kf]["arr"],
                          data_to_fit["seq_iops_train_" + n_kf]["arr"])

                elif "seq_lat" in k:
                    print("Modeling seq_iops")
                    v.fit(data_to_fit["seq_x_train_" + n_kf]["arr"],
                          data_to_fit["seq_lat_train_" + n_kf]["arr"])

        if not self.configs.cross_validation:  # or self.configs.visualization

            print("Model fitting without Cross Validation (CV).")

            data_to_fit = self.get_data()["train_data"]
            print("data_to_fit:", data_to_fit["rnd_x"]["arr"].shape)

            if not models_kf:
                print("Building models without CV.")
                models_kf = self.build_kf_models(n_folds=1)

            for k, v in tqdm(models_kf.items()):

                if "rnd_iops" in k:
                    print("Modeling rnd_iops",)
                    v.fit(data_to_fit["rnd_x"]["arr"],
                          data_to_fit["rnd_iops"]["arr"])

                elif "rnd_lat" in k:
                    print("Modeling rnd_lat")
                    v.fit(data_to_fit["rnd_x"]["arr"],
                          data_to_fit["rnd_lat"]["arr"])

                elif "seq_iops" in k:
                    print("Modeling seq_iops")
                    v.fit(data_to_fit["seq_x"]["arr"],
                          data_to_fit["seq_iops"]["arr"])

                elif "seq_lat" in k:
                        print("Modeling seq_iops")
                        v.fit(data_to_fit["seq_x"]["arr"],
                              data_to_fit["seq_lat"]["arr"])

        return models_kf

    def predict(self, models=None):

        # For visualization in the case of using CV for building models,
        # we need to pass model via predict argument, o.w. remain it unchanged.
        if not models:
            models = self.models

        if self.configs.cross_validation:
            print("Cross validation is being applied for prediction")
            data_to_pred = self.grouped_data_kf_pp  # self.get_data()

            for k, v in tqdm(models.items()):
                n_kf = str(k.split("_")[-1])  # number of fold

                if "rnd_iops" in k:
                    # print("Predicting rnd_iops")
                    self.predictions[k] = v.predict(data_to_pred["rnd_x_test_" + n_kf]["arr"], )

                elif "rnd_lat" in k:
                    # print("Predicting rnd_lat")
                    self.predictions[k] = v.predict(data_to_pred["rnd_x_test_" + n_kf]["arr"], )

                elif "seq_iops" in k:
                    # print("Predicting seq_iops")
                    self.predictions[k] = v.predict(data_to_pred["seq_x_test_" + n_kf]["arr"], )

                elif "seq_lat" in k:
                    # print("Predicting seq_iops")
                    self.predictions[k] = v.predict(data_to_pred["seq_x_test_" + n_kf]["arr"], )

        else:
            print("No cross validation for prediction")
            data_to_pred = self.grouped_train_test_pp["test_data"]  # self.get_data()["test_data"]

            for k, v in tqdm(models.items()):

                if "rnd_iops" in k:
                    print("Predicting rnd_iops")
                    self.predictions[k] = v.predict(data_to_pred["rnd_x"]["arr"], )

                elif "rnd_lat" in k:
                    print("Predicting rnd_lat")
                    self.predictions[k] = v.predict(data_to_pred["rnd_x"]["arr"], )

                elif "seq_iops" in k:
                    print("Predicting seq_iops")
                    self.predictions[k] = v.predict(data_to_pred["seq_x"]["arr"], )

                elif "seq_lat" in k:
                    print("Predicting seq_iops")
                    self.predictions[k] = v.predict(data_to_pred["seq_x"]["arr"], )

        return self.predictions

    def evaluate(self, ):

        if self.configs.cross_validation:
            print("Cross validation is being applied for evaluation")
            data_to_use = self.grouped_data_kf_pp  # self.get_data()
            for k, v in tqdm(self.predictions.items()):
                n_kf = str(k.split("_")[-1])  # number of fold
                print("k:", k,)
                if "rnd_iops" in k:
                    key = "rnd_iops"
                    if not self.results[key]:
                        self.results[key] = {}
                        self.results[key]["gb_mu"] = []
                        self.results[key]["qda_mu"] = []
                        self.results[key]["meape_mu"] = []

                    meape_mu, gb_mu, qda_mu = metrics.evaluate_a_x_test(
                        y_trues=data_to_use[key+"_test_"+n_kf]["arr"],
                        y_preds=v
                    )

                    self.results[key]["gb_mu"].append(gb_mu)
                    self.results[key]["qda_mu"].append(qda_mu)
                    self.results[key]["meape_mu"].append(meape_mu)

                elif "rnd_lat" in k:
                    key = "rnd_lat"
                    if not self.results[key]:
                        self.results[key] = {}
                        self.results[key]["gb_mu"] = []
                        self.results[key]["qda_mu"] = []
                        self.results[key]["meape_mu"] = []

                    meape_mu, gb_mu, qda_mu = metrics.evaluate_a_x_test(
                        y_trues=data_to_use[key+"_test_"+n_kf]["arr"],
                        y_preds=v
                    )

                    self.results[key]["gb_mu"].append(gb_mu)
                    self.results[key]["qda_mu"].append(qda_mu)
                    self.results[key]["meape_mu"].append(meape_mu)

                elif "seq_iops" in k:
                    key = "seq_iops"
                    if not self.results[key]:
                        self.results[key] = {}
                        self.results[key]["gb_mu"] = []
                        self.results[key]["qda_mu"] = []
                        self.results[key]["meape_mu"] = []

                    meape_mu, gb_mu, qda_mu = metrics.evaluate_a_x_test(
                        y_trues=data_to_use[key+"_test_"+n_kf]["arr"],
                        y_preds=v
                    )

                    self.results[key]["gb_mu"].append(gb_mu)
                    self.results[key]["qda_mu"].append(qda_mu)
                    self.results[key]["meape_mu"].append(meape_mu)

                elif "seq_lat" in k:
                    key = "seq_lat"
                    if not self.results[key]:
                        self.results[key] = {}
                        self.results[key]["gb_mu"] = []
                        self.results[key]["qda_mu"] = []
                        self.results[key]["meape_mu"] = []

                    meape_mu, gb_mu, qda_mu = metrics.evaluate_a_x_test(
                        y_trues=data_to_use[key+"_test_"+n_kf]["arr"],
                        y_preds=v
                    )

                    self.results[key]["gb_mu"].append(gb_mu)
                    self.results[key]["qda_mu"].append(qda_mu)
                    self.results[key]["meape_mu"].append(meape_mu)

        else:
            print("No cross validation for evaluation")
            data_to_use = self.grouped_train_test_pp["test_data"]  # self.get_data()["test_data"]

            for k, v in tqdm(self.predictions.items()):
                n_kf = str(k.split("_")[-1])  # number of fold
                print("k:", k,)

                if "rnd_iops" in k:
                    key = "rnd_iops"
                    if not self.results[key]:
                        self.results[key] = {}
                        self.results[key]["gb_mu"] = []
                        self.results[key]["qda_mu"] = []
                        self.results[key]["meape_mu"] = []

                    meape_mu, gb_mu, qda_mu = metrics.evaluate_a_x_test(
                        y_trues=data_to_use[key]["arr"],
                        y_preds=v
                    )

                    self.results[key]["gb_mu"].append(gb_mu)
                    self.results[key]["qda_mu"].append(qda_mu)
                    self.results[key]["meape_mu"].append(meape_mu)

                elif "rnd_lat" in k:
                    key = "rnd_lat"
                    if not self.results[key]:
                        self.results[key] = {}
                        self.results[key]["gb_mu"] = []
                        self.results[key]["qda_mu"] = []
                        self.results[key]["meape_mu"] = []

                    meape_mu, gb_mu, qda_mu = metrics.evaluate_a_x_test(
                        y_trues=data_to_use[key]["arr"],
                        y_preds=v
                    )

                    self.results[key]["gb_mu"].append(gb_mu)
                    self.results[key]["qda_mu"].append(qda_mu)
                    self.results[key]["meape_mu"].append(meape_mu)

                elif "seq_iops" in k:
                    key = "seq_iops"
                    if not self.results[key]:
                        self.results[key] = {}
                        self.results[key]["gb_mu"] = []
                        self.results[key]["qda_mu"] = []
                        self.results[key]["meape_mu"] = []

                    meape_mu, gb_mu, qda_mu = metrics.evaluate_a_x_test(
                        y_trues=data_to_use[key]["arr"],
                        y_preds=v
                    )

                    self.results[key]["gb_mu"].append(gb_mu)
                    self.results[key]["qda_mu"].append(qda_mu)
                    self.results[key]["meape_mu"].append(meape_mu)

                elif "seq_lat" in k:
                    key = "seq_lat"
                    if not self.results[key]:
                        self.results[key] = {}
                        self.results[key]["gb_mu"] = []
                        self.results[key]["qda_mu"] = []
                        self.results[key]["meape_mu"] = []

                    meape_mu, gb_mu, qda_mu = metrics.evaluate_a_x_test(
                        y_trues=data_to_use[key]["arr"],
                        y_preds=v
                    )

                    self.results[key]["gb_mu"].append(gb_mu)
                    self.results[key]["qda_mu"].append(qda_mu)
                    self.results[key]["meape_mu"].append(meape_mu)

        return self.results

    def visualize_results(self, cross_val_num: int=1):  #  models, predictions,

        cross_val_num = str(cross_val_num)
        vd_all = [d for d in self.visualization_data]
        vd_all = pd.concat(vd_all)
        vd_all_grouped = self.data.get_grouped_data(data=vd_all)

        pp_vd_all = PreProcess(
            data=vd_all_grouped,
            out_dist="uniform",
        )

        vd_all_grouped_pp = defaultdict(list)
        vd_all_grouped_pp["all_visual_data"] = pp_vd_all.pre_process()

        print("vd_all_grouped_pp:", vd_all_grouped_pp)


        for a_data in self.visualization_data:
            a_grouped_data = self.data.get_grouped_data(data=a_data)

            # print("a_grouped_data: \n",
            #       a_grouped_data
            #       )

            _rnd_x = a_grouped_data["rnd_x"].values
            rnd_lat = a_grouped_data["rnd_lat"].values.reshape(-1, 1)
            rnd_iops = a_grouped_data["rnd_iops"].values.reshape(-1, 1)


            _seq_x = a_grouped_data["seq_x"].values
            seq_lat = a_grouped_data["seq_lat"].values.reshape(-1, 1)
            seq_iops = a_grouped_data["seq_iops"].values.reshape(-1, 1)

            if len(_rnd_x) != 0:

                # pre-process the visualization data
                qt_rnd_x = vd_all_grouped_pp["all_visual_data"]["rnd_x"]["_qt"]
                qt_rnd_iops = vd_all_grouped_pp["all_visual_data"]["rnd_iops"]["_qt"]
                qt_rnd_lat = vd_all_grouped_pp["all_visual_data"]["rnd_lat"]["_qt"]

                rnd_x = pp_vd_all._fit_quantile_standardizer(qt_rnd_x, _rnd_x)
                rnd_iops = pp_vd_all._fit_quantile_standardizer(qt_rnd_iops, rnd_iops)
                rnd_lat = pp_vd_all._fit_quantile_standardizer(qt_rnd_lat, rnd_lat)

                iops_rnd_pred = self.models["rnd_iops_"+cross_val_num].predict(rnd_x).reshape(-1, 1)
                lat_rnd_pred = self.models["rnd_lat_"+cross_val_num].predict(rnd_x).reshape(-1, 1)
                y_preds_rnd = np.concatenate((iops_rnd_pred, lat_rnd_pred), axis=1, )
                y_trues_rnd = np.concatenate((rnd_iops, rnd_lat), axis=1, )

                title = "Write"
                if np.unique(rnd_x[:, 3])[0] == 0:
                    title = "Read"

                file_title_rnd = "".join(
                    "IO_depth" + str(np.unique(_rnd_x[:, 0])[0]) + \
                    "Block_size" + str(np.unique(_rnd_x[:, 1])[0]) + \
                    "RW" + str(np.unique(_rnd_x[:, 2])[0]) + \
                    "io_type" + str(np.unique(_rnd_x[:, 3])[0]) + \
                    "load_type" + str(np.unique(_rnd_x[:, 4])[0])
                )

                # print("file_title:", file_title_rnd, "\n",
                #       "report path:", self.configs.report_path
                #       )

                utils.basic_plots(y_true=y_trues_rnd,
                                  y_pred=y_preds_rnd,
                                  title=title,
                                  save_path=self.configs.report_path,
                                  file_title=file_title_rnd
                                  )

            if len(_seq_x) != 0 :

                # pre-process the visualization data
                qt_seq_x = vd_all_grouped_pp["all_visual_data"]["seq_x"]["_qt"]
                qt_seq_iops = vd_all_grouped_pp["all_visual_data"]["seq_iops"]["_qt"]
                qt_seq_lat = vd_all_grouped_pp["all_visual_data"]["seq_lat"]["_qt"]

                seq_x = pp_vd_all._fit_quantile_standardizer(qt_seq_x, _seq_x)
                seq_iops = pp_vd_all._fit_quantile_standardizer(qt_seq_iops, seq_iops)
                seq_lat = pp_vd_all._fit_quantile_standardizer(qt_seq_lat, seq_lat)

                iops_seq_pred = self.models["seq_iops_" + cross_val_num].predict(seq_x).reshape(-1, 1)
                lat_seq_pred = self.models["seq_lat_" + cross_val_num].predict(seq_x).reshape(-1, 1)
                y_preds_seq = np.concatenate((iops_seq_pred, lat_seq_pred), axis=1, )
                y_trues_seq = np.concatenate((seq_iops, seq_lat), axis=1, )

                title = "Write"
                if np.unique(seq_x[:, 3])[0] == 0:
                    title = "Read"

                file_title_seq = "".join(
                    "IO_depth" + str(np.unique(_seq_x[:, 0])[0]) + \
                    "Block_size" + str(np.unique(_seq_x[:, 1])[0]) + \
                    "RW" + str(np.unique(_seq_x[:, 2])[0]) + \
                    "io_type" + str(np.unique(_seq_x[:, 3])[0]) + \
                    "load_type" + str(np.unique(_seq_x[:, 4])[0])
                )

                # print("file_title:", file_title_seq, "\n",
                #       "report path:", self.configs.report_path
                #       )

                utils.basic_plots(y_true=y_trues_seq,
                                  y_pred=y_preds_seq,
                                  title=title,
                                  save_path=self.configs.report_path,
                                  file_title=file_title_seq
                                  )




# for k, v in self.grouped_data_kf_pp.items():
    #    try:
    #         print(k, v["arr"].shape)
    #         print("pp:", v["arr"].shape)
    #         print("k:", k, v["_qt"])
    #         print("rvt:", k, grouped_data_kf_pp_revert[k].shape)
    #         print("rvt:", k, grouped_data_kf_pp_revert[k])
    #    except:
    #        print("except:", k, v)

    # Train and test splits without KFolds
    # Train data

 # for k, v in self.grouped_train_test_pp["test_data"].items():
        #     try:
        #         print(k, v["arr"].shape)
        #         print("pp:", v["arr"].shape)
        #         print("k:", k, v["_qt"])
        #         print("rvt:", k, self.grouped_train_test_pp["test_data"][k].shape)
        #         print("rvt:", k, self.grouped_train_test_pp["test_data"][k])
        #     except:
        #         print("except:", k, v)

# for v in self.visualization_data:
        #     print(v)