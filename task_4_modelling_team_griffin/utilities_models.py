# imports.
import numpy as np

# ML imports.
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Data.
from utilities_features import FEATURES


# Baseline.
def make_baseline_train_fn(feature: str, direction: str, quantile: float):
    def train_fn(X, y, w):
        # "тренировка" = вычислить квантильный порог на train.
        idx = FEATURES.index(feature)
        threshold = np.nanquantile(X[:, idx], quantile)
        return {"feature_idx": idx, "threshold": threshold, "direction": direction}
    return train_fn

def make_baseline_predict_fn():
    def predict_fn(bundle, X):
        col = X[:, bundle["feature_idx"]]
        # возвращаем toxicity-like score, чтобы вписаться в общий пайплайн.
        return col if bundle["direction"] == "above" else -col
    return predict_fn


# Ridge model with weighted MSE.
def make_ridge_train_fn(alpha: float = 1.0):
    """
    Returns train_fn for Ridge regression.
    """
    def train_fn(X, y, w):

        # impute NaN -> 0, then standardize.
        X_clean = np.nan_to_num(X, nan=0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y, sample_weight=w)

        return {"model": model, "scaler": scaler}

    return train_fn

def make_ridge_predict_fn():
    """
    Returns predict_fn for Ridge regression.
    """
    def predict_fn(bundle, X):
        X_clean  = np.nan_to_num(X, nan=0.0)
        X_scaled = bundle["scaler"].transform(X_clean)
        return bundle["model"].predict(X_scaled).astype(np.float32)
    
    return predict_fn


# Elastic net + huber loss.
def make_huber_elasticnet_train_fn(
    alpha: float = 1e-4,
    l1_ratio: float = 0.15,
    epsilon: float = 1.35,
    max_iter: int = 2000,
    tol: float = 1e-4
):
    """
    Returns train_fn for Linear model with Huber loss and ElasticNet regularization.

    Loss:
        Huber(y, y_pred)

    Regularization:
        alpha * [l1_ratio * L1 + (1 - l1_ratio) * L2]

    Parameters:
        alpha:
            Total regularization strength.

        l1_ratio:
            Balance between L1 and L2.
            l1_ratio=0.0 -> pure L2
            l1_ratio=1.0 -> pure L1
            l1_ratio=0.15 -> mostly L2 with some sparsity.

        epsilon:
            Huber threshold.
            Smaller epsilon -> more robust, more aggressive clipping of outliers.
    """

    def train_fn(X, y, w):

        X_clean = np.nan_to_num(X, nan=0.0).astype(np.float32)

        y = np.asarray(y, dtype=np.float32)
        w = np.asarray(w, dtype=np.float32)
        w = w / w.mean()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)


        model = SGDRegressor(
            loss            = "huber",
            penalty         = "elasticnet",
            alpha           = alpha,
            l1_ratio        = l1_ratio,
            epsilon         = epsilon,
            max_iter        = max_iter,
            tol             = tol,
            random_state    = 42,
            learning_rate   = "adaptive",
            eta0            = 0.01,
            average         = True
        )

        model.fit(X_scaled, y, sample_weight=w)

        return {
            "model": model,
            "scaler": scaler,
        }

    return train_fn

def make_huber_elasticnet_predict_fn():
    """
    Returns predict_fn for Linear model with Huber loss and ElasticNet regularization.
    """

    def predict_fn(bundle, X):
        X_clean = np.nan_to_num(X, nan=0.0).astype(np.float32)

        X_scaled = bundle["scaler"].transform(X_clean)

        return bundle["model"].predict(X_scaled).astype(np.float32)

    return predict_fn


# LightGBM with weighted MSE.
def make_lgbm_train_fn(
    max_depth:          int = -1,
    num_leaves:         int = 63,
    learning_rate:      float = 0.05,
    n_estimators:       int = 1000,
    min_child_samples:  int = 500,
    reg_lambda:         float = 1.0,
    subsample:          float = 0.8,
    colsample_bytree:   float = 0.8,
    early_stopping_rounds:  int = 50,
    verbose:                int = -1,
):
    """
    Returns train_fn for LightGBM regressor with weighted MSE.
    """

    def train_fn(X, y, w):

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        w = np.asarray(w, dtype=np.float32)

        n = y.size
        split_idx = int(n * 0.85)

        X_tr, X_val = X[:split_idx], X[split_idx:]
        y_tr, y_val = y[:split_idx], y[split_idx:]
        w_tr, w_val = w[:split_idx], w[split_idx:]

        model = lgb.LGBMRegressor(
            max_depth           = max_depth,
            objective           = "regression",
            num_leaves          = num_leaves,
            learning_rate       = learning_rate,
            n_estimators        = n_estimators,
            min_child_samples   = min_child_samples,
            reg_lambda          = reg_lambda,
            subsample           = subsample,
            colsample_bytree    = colsample_bytree,
            random_state        = 42,
            n_jobs              = -1,
            verbose             = verbose
        )

        model.fit(
            X_tr, y_tr,
            sample_weight       = w_tr,
            eval_set            = [(X_val, y_val)],
            eval_sample_weight  = [w_val],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=verbose > 0),
                lgb.log_evaluation(period=0),
            ],
        )

        return model

    return train_fn

def make_lgbm_predict_fn():
    """
    Returns predict_fn for LightGBM regressor with weighted MSE.
    """

    def predict_fn(model, X):
        X = np.asarray(X, dtype=np.float32)
        return model.predict(X).astype(np.float32)

    return predict_fn


# LightGBM with Huber loss.
def make_lgbm_huber_train_fn(
    max_depth:          int = -1,
    huber_alpha:        float = 0.9,                
    num_leaves:         int = 63,
    learning_rate:      float = 0.05,
    n_estimators:       int = 1000,
    min_child_samples:  int = 500,
    reg_lambda:         float = 1.0,
    subsample:          float = 0.8,
    colsample_bytree:   float = 0.8,
    early_stopping_rounds:  int = 50,
    verbose:                int = -1,
):
    """
    LightGBM with Huber loss. Robust to pnl tail outliers.

    Parameters
    ----------
    huber_alpha : float
        Huber's delta parameter. LightGBM uses 'alpha' as the delta cutoff.
        Smaller -> more L1-like (robust). Larger → more L2-like.
    """
    def train_fn(X, y, w):
        import lightgbm as lgb

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        w = np.asarray(w, dtype=np.float32)

        n = len(y)
        split_idx = int(n * 0.85)
        X_tr, X_val = X[:split_idx], X[split_idx:]
        y_tr, y_val = y[:split_idx], y[split_idx:]
        w_tr, w_val = w[:split_idx], w[split_idx:]

        model = lgb.LGBMRegressor(
            max_depth           = max_depth,
            objective           = "huber",
            alpha               = huber_alpha,
            num_leaves          = num_leaves,
            learning_rate       = learning_rate,
            n_estimators        = n_estimators,
            min_child_samples   = min_child_samples,
            reg_lambda          = reg_lambda,
            subsample           = subsample,
            colsample_bytree    = colsample_bytree,
            random_state        = 42,
            n_jobs              = -1,
            verbose             = verbose
        )

        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=verbose > 0),
                lgb.log_evaluation(period=0),
            ],
        )
        return model

    return train_fn



# Custom Models Setups.
CUSTOM_MODELS = {

    "RuleBased_DistToMid": {
        "train_fn":         make_baseline_train_fn(
            feature   = "dist_to_mid",
            direction = "above",
            quantile  = 0.90
        ),
        "predict_fn":       make_baseline_predict_fn(),
        "prediction_type":  "toxicity",
        "description":      "Best rule-based filter from Task 2/3 (dist-to-mid threshold)."
    },

    "Ridge": {
        "train_fn":  make_ridge_train_fn(alpha=1.0),
        "predict_fn": make_ridge_predict_fn(),
        "threshold_direction": "below",   # filter where predicted pnl is low.
        "description": "Ridge regression with weighted MSE on pnl.",
    },

    "Linear_Huber_ElasticNet": {
        "train_fn": make_huber_elasticnet_train_fn(
            alpha       = 1e-4,
            l1_ratio    = 0.15,
            epsilon     = 1.35,
            max_iter    = 200,
            tol         = 1e-4
        ),
        "predict_fn": make_huber_elasticnet_predict_fn(),
        "threshold_direction": "below",
        "description": "Linear regressor with Huber loss and ElasticNet L1+L2 regularization on pnl.",
    },

    "LightGBM": {
        "train_fn":  make_lgbm_train_fn(),
        "predict_fn": make_lgbm_predict_fn(),
        "threshold_direction": "below",
        "description": "LightGBM regressor with weighted MSE on pnl.",
    },

    "LGBM_Huber": {
        "train_fn":   make_lgbm_huber_train_fn(huber_alpha=0.9),
        "predict_fn": make_lgbm_predict_fn(),
        "prediction_type": "pnl",
        "description": "LightGBM with Huber loss.",
    }
}
