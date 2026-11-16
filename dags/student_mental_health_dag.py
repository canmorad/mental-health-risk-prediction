from datetime import datetime
from airflow import DAG
from airflow.decorators import task

from src.preprocessing import load_and_preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

with DAG(
    dag_id="student_mental_health_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    @task
    def run_ml_pipeline():
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("/opt/airflow/data/student_mental_health.csv")
        model = train_model(X_train, y_train, preprocessor, "/opt/airflow/models/model_dag.pkl")
        metrics = evaluate_model(X_test, y_test, model)
        return metrics

    run_ml_pipeline()