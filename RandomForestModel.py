from Model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

class RandomForestModel(Model):
    def __init__(self):
        self.param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt'],
            'class_weight': ['balanced']
        }

    def use_grid_search(self, representation: np.array, labels: np.array):
        """
        Trains RandomForestClassifier using GridSearchCV to find the best hyperparameters.

        Returns:
            tuple: (best_estimator, best_params)
        """
        rf = RandomForestClassifier(random_state=42)
        grid_search_rf = GridSearchCV(
            estimator=rf,
            param_grid=self.param_grid_rf,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='f1_macro'
        )
        grid_search_rf.fit(representation, labels)
        print(f"Best hyperparameters: {grid_search_rf.best_params_}")
        return grid_search_rf.best_estimator_, grid_search_rf.best_params_

    def create_model(self, **best_params: dict):
        """
        Creates a RandomForestClassifier model with the best hyperparameters.

        Returns:
            RandomForestClassifier: model
        """
        rf = RandomForestClassifier(random_state=42)
        rf.set_params(**best_params)
        return rf
