import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def mean_estimation_absolute_percentage_error(y_true, y_pred, n_iters=100):

    errors = []
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)

    inds = np.arange(len(y_true))
    for i in range(n_iters):
        inds_boot = resample(inds)

        y_true_boot = y_true[inds_boot]
        y_pred_boot = y_pred[inds_boot]

        y_true_mean = y_true_boot.mean(axis=0)
        y_pred_mean = y_pred_boot.mean(axis=0)

        ierr = np.abs((y_true_mean - y_pred_mean) / y_true_mean) * 100
        errors.append(ierr)

    errors = np.array(errors)
    return errors.mean(axis=0), errors.std(axis=0)


def discrepancy_score(observations, forecasts, model="QDA", n_iters=1):
    """
    Parameters:
    -----------
    observations : numpy.ndarray, shape=(n_samples, n_features)
        True values.
        Example: [[1, 2], [3, 4], [4, 5], ...]
    forecasts : numpy.ndarray, shape=(n_samples, n_features)
        Predicted values.
        Example: [[1, 2], [3, 4], [4, 5], ...]
    model : sklearn binary classifier
        Possible values: RF, DT, LR, QDA, GBDT
    n_iters : int
        Number of iteration per one forecast.

    Returns:
    --------
    mean : float
        Mean value of discrepancy score.
    std : float
        Standard deviation of the mean discrepancy score.

    """

    observations = np.asarray(observations).reshape(-1, 1)
    forecasts = np.asarray(forecasts).reshape(-1, 1)
    eps = np.random.normal(loc=0, scale=10 ** -6, size=forecasts.shape)
    forecasts = forecasts + eps  # to avoid error when we use QDA

    scores = []

    X0 = observations
    y0 = np.zeros(len(observations))

    X1 = forecasts
    y1 = np.ones(len(forecasts))

    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((y0, y1), axis=0)

    for it in range(n_iters):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, shuffle=True
        )
        if model == "RF":
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=10, max_features=None
            )
        elif model == "GDBT":
            clf = GradientBoostingClassifier(max_depth=6, subsample=0.7)
        elif model == "DT":
            clf = DecisionTreeClassifier(max_depth=10)
        elif model == "LR":
            clf = LogisticRegression()
        elif model == "QDA":
            clf = QuadraticDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict_proba(X_test)[:, 1]
        auc = 2 * roc_auc_score(y_test, y_pred_test) - 1
        scores.append(auc)

    scores = np.array(scores)
    mean = scores.mean()
    std = scores.std() / np.sqrt(len(scores))

    return mean, std


def evaluate_a_x_test(y_trues, y_preds,):

    # MEAPE >> Mean Estimation Absolute Percentage Error
    meape_mu, meape_std = mean_estimation_absolute_percentage_error(y_trues, y_preds, n_iters=100)

    # gb >> Gradient Decent Boosting Classifier
    gb_mu, gb_std = discrepancy_score(y_trues, y_preds, model='GDBT', n_iters=10)

    # qda >> Quadratic Discriminant Analysis
    qda_mu, qda_std = discrepancy_score(y_trues, y_preds, model='QDA', n_iters=10)

    return meape_mu, gb_mu, qda_mu


