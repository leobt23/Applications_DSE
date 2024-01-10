import os

import matplotlib.pyplot as plt
import pandas as pd
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
        }.get(strategy_name, SMOTE)

    def save_plot(self, df, y, file_path, pre_post_flag, strategy_name):
        # df numpy array to pandas dataframe
        column_names = [
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",
            "Amount",
            "hour",
            "Class",
        ]
        # df and y are numpy arrays so we need to convert them to pandas dataframe
        df = pd.DataFrame(df, columns=column_names[:-1])
        y = pd.DataFrame(y, columns=["Class"])

        # join X and y
        df = pd.concat([df, y], axis=1)
        plt.figure(figsize=(10, 6))  # You can adjust the figure size
        for class_value in df["Class"].unique():
            plt.scatter(
                df[df["Class"] == class_value]["V17"],
                df[df["Class"] == class_value]["V14"],
                label=class_value,
            )

        # Customize the plot
        plt.title("Oversampled by SMOTE", color="black")
        plt.xlabel("V17", color="black")
        plt.ylabel("V14", color="black")
        plt.legend(title="Class", title_fontsize="13", fontsize="10")
        plt.grid(True)
        plt.gca().set_facecolor("white")

        file_path = f"data_generated/resampling/"

        # create the file path if it doesn't exist
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        _file_path = f"{file_path}{pre_post_flag}_{strategy_name}.png"

        plt.savefig(_file_path)

    def apply_resampling(self, strategy_name, X, y, sampling_strategy="auto"):
        # Create a new X and add y to it
        self.save_plot(X, y, "data_generated/resampling/", "pre", strategy_name)
        strategy_class = self.get_resampling_class(strategy_name)
        resampler = strategy_class(sampling_strategy=sampling_strategy)
        X_res, y_res = resampler.fit_resample(X, y)
        app_logger.info(
            f"Resampling applied: {strategy_name} with sampling_strategy={sampling_strategy}"
        )
        self.save_plot(
            X_res, y_res, "data_generated/resampling/", "post", strategy_name
        )
        return X_res, y_res
