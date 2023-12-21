from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    AllKNN,
    ClusterCentroids,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)

from src.data.abstract_resampling import Resampling
from src.logger_cfg import app_logger


class Resampler(Resampling):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod  # Static method to be able to call it without instantiating the class
    def get_resampling_class(strategy_name):
        return {
            "SMOTE": SMOTE,
            "ADASYN": ADASYN,
            "SVMSMOTE": SVMSMOTE,
            "BorderlineSMOTE": BorderlineSMOTE,
            "KMeansSMOTE": KMeansSMOTE,
            "RandomOverSampler": RandomOverSampler,
            "AllKNN": AllKNN,
            "ClusterCentroids": ClusterCentroids,
            "CondensedNearestNeighbour": CondensedNearestNeighbour,
            "EditedNearestNeighbours": EditedNearestNeighbours,
            "InstanceHardnessThreshold": InstanceHardnessThreshold,
            "NearMiss": NearMiss,
            "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule,
            "OneSidedSelection": OneSidedSelection,
            "RandomUnderSampler": RandomUnderSampler,
            "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours,
            "TomekLinks": TomekLinks,
        }.get(
            strategy_name, SMOTE
        )  # Default to SMOTE if not found

    def apply_resampling(self, strategy_name, X, y, sampling_strategy="auto"):
        strategy_class = self.get_resampling_class(strategy_name)
        resampler = strategy_class(sampling_strategy=sampling_strategy)
        X_res, y_res = resampler.fit_resample(X, y)
        app_logger.info(
            f"Resampling applied: {strategy_name} with sampling_strategy={sampling_strategy}"
        )
        return X_res, y_res
