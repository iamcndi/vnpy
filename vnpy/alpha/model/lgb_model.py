"""
LightGBM Alpha 因子模型

可跨脚本序列化加载：训练时用 AlphaLab.save_model() 保存，
预测时用 AlphaLab.load_model() 加载。
"""

from datetime import datetime

import lightgbm as lgb
import numpy as np
import polars as pl

from vnpy.alpha.model.template import AlphaModel
from vnpy.alpha.dataset import AlphaDataset, Segment


class LGBAlphaModel(AlphaModel):
    """LightGBM Alpha 因子模型"""

    def __init__(self, params: dict | None = None) -> None:
        self.model: lgb.LGBMRegressor | None = None
        self.feature_cols: list[str] = []

        self.params: dict = params or {
            "objective": "regression",
            "metric": "mse",
            "num_leaves": 24,
            "learning_rate": 0.03,
            "n_estimators": 800,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbosity": -1,
        }

    def fit(self, dataset: AlphaDataset) -> None:
        """训练模型"""
        train_feat = dataset.fetch_raw(Segment.TRAIN)
        self.feature_cols = [c for c in train_feat.columns
                             if c not in ("datetime", "vt_symbol", "label")]

        train_start, train_end = dataset.data_periods[Segment.TRAIN]
        result = dataset.result_df.select(["datetime", "vt_symbol", "label"])

        train_labels = result.filter(
            pl.col("datetime") >= pl.lit(datetime.strptime(train_start, "%Y-%m-%d")),
        ).filter(
            pl.col("datetime") <= pl.lit(datetime.strptime(train_end, "%Y-%m-%d")),
        )

        train_data = train_feat.join(train_labels, on=["datetime", "vt_symbol"], how="inner")

        X = train_data.select(self.feature_cols).to_numpy()
        y = train_data["label"].to_numpy()

        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]
        y = np.clip(y, -0.15, 0.15)

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y, eval_set=[(X, y)], eval_metric="mse")

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """生成预测信号"""
        feat = dataset.fetch_raw(segment)
        X = feat.select(self.feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X)
