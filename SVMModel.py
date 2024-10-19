from Model import Model
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVMModel(Model):
    def __init__(self):
        self.param_grid = {
            'C': [0.1, 1.0, 10],
            'class_weight': ['balanced'],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 1, 0.1, 0.01]
        }

    def use_grid_search(self, representation, labels):
        svm = SVC(random_state=42)
        grid_search_svm = GridSearchCV(
            estimator=svm,
            param_grid=self.param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='f1_macro'
        )
        grid_search_svm.fit(representation, labels)
        print(f"Best hyperparameters: {grid_search_svm.best_params_}")
        return grid_search_svm.best_estimator_, grid_search_svm.best_params_
    
    def create_model(self, **best_params):
        svm = SVC(random_state=42)
        svm.set_params(**best_params)
        return svm