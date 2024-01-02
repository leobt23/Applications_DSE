from typing import Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.abstract_model_tester import AbstractModelTester
from src.utils import app_logger


class NNModelTester(AbstractModelTester):
    pass
