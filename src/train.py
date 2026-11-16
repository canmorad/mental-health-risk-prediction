from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X_train, y_train, preprocessor, path_save_model):
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    C=1,
                    class_weight={"High": 1.0, "Low": 1.0, "Medium": 1.1},
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    joblib.dump(model, path_save_model)
    return model