from pathlib import Path
from ssd_sim.models.estimator import XGBoostReg
from ssd_sim.common.utils import print_predictions, print_evaluated_results


from ssd_sim.data.ssd_data import SSDDATA
# from ssd_sim.data.preprocess import PreProcess

root_dir = Path("/home/Soroosh/tmpYadro/data-storage/data/prod-dataset/")  # change correspondingly

paths = (
    root_dir / "hse-fio__disk-type=ssd__load_type=seq__2021-10-22_14-19-31/",
    root_dir / "hse-fio__disk-type=ssd__load_type=rnd__2021-10-22_14-12-08/",
)

configs = {
    "learning_rate": 1e-2,
    "max_depth": 5,
    "n_estimators": 10000,
    "objective": "reg:squarederror",
    "lambda": 1.2,
    "cross_validation": False,
    "n_folds": 2,
    "paths": paths,
    "logging": False,
    "save_path": Path("/home/Soroosh/ssd-simulation/Savings/"),
    "report_path": Path("/home/Soroosh/ssd-simulation/Reports/"),
    "use_last_n_test_split": 50,
}

# "model_params": {"n_estimators": 1000, "max_depth": 7, },  # n_estimators=10000

if not configs["cross_validation"]:
    configs["visualization"] = True
else:
    configs["visualization"] = True
    print("At the moment visualization is not supported for KFold Cross validation ")

if __name__ == "__main__":

    # assert configs.visualization is True and

    # prediction = {}

    regressor = XGBoostReg(configs)

    models = regressor.fit_models()
    predictions = regressor.predict()
    results_evaluated = regressor.evaluate()
    print_predictions(predictions)
    print_evaluated_results(results_evaluated)
    regressor.visualize_results(cross_val_num=1)

    print("Finish!")

