from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(X_test, y_test, model):
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted", zero_division=0)
    recall = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return metrics