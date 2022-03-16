import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ssd_sim.common import metrics


def basic_plots(y_true, y_pred, title, save_path, file_title, ):

    iops_true = y_true[:, 0]
    iops_pred = y_pred[:, 0]

    lat_true = y_true[:, 1]  # / 10 ** 6  # bcs I don't inverse the predictions
    lat_pred = y_pred[:, 1]  # / 10 ** 6  # bcs I don't inverse the predictions

    plt.figure(figsize=(21, 6))

    plt.subplot(131)
    plt.scatter(iops_true, lat_true, alpha=0.5, marker='o', label='True')
    plt.scatter(iops_pred, lat_pred, alpha=0.5, marker='+', label='Prediction')
    plt.xlabel('IOPS', size=14)
    plt.ylabel('Latency, ms', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(title, size=14)
    plt.legend(loc='best', fontsize=10)


    plt.subplot(132)
    vals = np.concatenate((iops_true, iops_pred))
    bins = np.linspace(vals.min(), vals.max(), 50)
    plt.hist(iops_true, bins=bins, alpha=1., label='True', histtype='step', linewidth=3)
    plt.hist(iops_pred, bins=bins, alpha=1., label='Prediction')
    plt.xlabel('IOPS', size=14)
    plt.ylabel('Counts', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(title, size=14)
    plt.legend(loc='best', fontsize=10)


    plt.subplot(133)
    vals = np.concatenate((lat_true, lat_pred))
    bins = np.linspace(vals.min(), vals.max(), 50)
    plt.hist(lat_true, bins=bins, alpha=1., label='True', histtype='step', linewidth=3)
    plt.hist(lat_pred, bins=bins, alpha=1., label='Prediction')
    plt.xlabel('Latency, ms', size=14)
    plt.ylabel('Counts', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(title, size=14)
    plt.legend(loc='best', fontsize=10)

    plt.savefig(os.path.join(save_path, file_title + ".png"))

    plt.show()


def print_predictions(predictions):

    for k, v in predictions.items():
        print("Predicting " + k + ": \n")
        print("\t", v)
    return None


def print_evaluated_results(results_evaluated):

    for k, v in results_evaluated.items():
        print("Evaluated results for", k)
        for kk, vv in v.items():
            res = np.asarray(vv)
            m = res.mean()
            s = res.std()
            print("\t", kk, "mean:", m, "std:", s)
