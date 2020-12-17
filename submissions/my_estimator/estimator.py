import numpy as np
import pandas as pd
import ruptures as rpt

from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator



class Ruptureestimator(BaseEstimator):
    def __init__(self):
        self.window_length = 19
        self.polyorder = 4
        self.rpt_min = 42

    def fit(self, X, y):
        return (self)

    def side_effect(
        self, data,
        t_init=3,
        t_final=1,
        coef_mult=0.05,
        freq=100,
    ):
        start = t_init * freq
        end = t_final * freq
        data[:start] = coef_mult*np.random.rand(start)
        data[-end:] = coef_mult*np.random.rand(end)
        return (data)

    def rpt_conv(self, L, margin=(0, 0)):
        min, max = margin
        LL = []
        for i in range(0, len(L)-2, 2):
            LL.append([L[i]-(min*(L[i+1]-L[i])), L[i+1]+(max*(L[i+1]-L[i]))])
        return(LL)

    def rpt_predict(
        self, data, min_size, pen=10, margin=(0.15, 0.1)
    ):
        algo = rpt.Pelt(model="rbf", min_size=min_size)
        fit = algo.fit(data)
        breakpt = fit.predict(pen=pen)
        return self.rpt_conv(breakpt, margin)

    def predict(self, X):
        y_pred = []

        # preprocessing
        train = pd.concat([signal.signal for signal in X])
        mean = train.mean()
        deviation = train.std()

        for signal in X:
            signal.signal = (signal.signal - mean) / deviation
            signal.signal['Max'] = signal.signal.max(axis=1)

        df_X = pd.DataFrame([vars(sig) for sig in X])

        # predict
        for i, signal in df_X.iterrows():
            data = signal['signal']
            data = data['Max'].to_numpy()

            data = self.side_effect(data)
            data = savgol_filter(
                data, self.window_length, self.polyorder
            )

            result = self.rpt_predict(
                data, self.rpt_min
            )
            y_pred += [result]
        return np.array(y_pred, dtype=list)


def get_estimator():
    detector = Ruptureestimator()
    return Pipeline(steps=[('detector', detector)])
