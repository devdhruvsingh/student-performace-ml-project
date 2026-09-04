import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path=None):
        try:
            logging.info("Split training and test input data")
            
            # Fix 1: Correctly slice test_array for test inputs
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "linear regression": LinearRegression(),
                "lasso": Lasso(),
                "ridge": Ridge(),
                "k-neighbors regressor": KNeighborsRegressor(),
                "decision tree": DecisionTreeRegressor(),
                "random forest regressor": RandomForestRegressor(),
                "adaboost regressor": AdaBoostRegressor(),
                "gradient boosting": GradientBoostingRegressor(),
                "xgbregressor": XGBRegressor(),
                "catboosting regressor": CatBoostRegressor(verbose=False),
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            # Get best score
            best_model_score = max(model_report.values())

            # Fix 2: Fetch model name string directly using max() key
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            # Fix 4: Pass sys to CustomException
            if best_model_score < 0.6:
                raise CustomException("No model found with score above threshold", sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Fix 3: Updated to match correct config property name
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)