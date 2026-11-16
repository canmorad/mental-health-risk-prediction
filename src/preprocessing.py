import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

nominal_cols = ["Gender", "Part_Time_Job", "Previous_Mental_Health_History"]
ordinal_cols = ["Physical_Activity", "Diet_Quality"]
numeric_cols = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]

def swap_values(row):
    if row["Social_Media_Hours"] > row["Screen_Time_Hours"]:
        row["Social_Media_Hours"], row["Screen_Time_Hours"] = (
            row["Screen_Time_Hours"],
            row["Social_Media_Hours"],
        )
    return row

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df[["Social_Media_Hours", "Screen_Time_Hours"]] = df[
        ["Social_Media_Hours", "Screen_Time_Hours"]
    ].apply(swap_values, axis=1)

    X = df.drop(columns=["Mental_Health_Risk", "Student_ID"])
    y = df["Mental_Health_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
    )

    nominal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[
                        ["Low", "Moderate", "High"],
                        ["Poor", "Average", "Good"],
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_cols),
            ("nom", nominal_pipeline, nominal_cols),
            ("ord", ordinal_pipeline, ordinal_cols),
        ]
    )
    return X_train, X_test, y_train, y_test, preprocessor