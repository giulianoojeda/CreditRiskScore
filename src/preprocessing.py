from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # Initialize encoders
    ordinal_encoder = OrdinalEncoder()
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Select the columns to encode
    object_cols = working_train_df.select_dtypes(include="object").columns

    # Get the ordinal columns
    ordinal_cols = [col for col in object_cols if working_train_df[col].nunique() == 2]
    # Get the one-hot columns
    one_hot_cols = [col for col in object_cols if working_train_df[col].nunique() > 2]

    # Fit the enconders only on the train data
    if ordinal_cols:
        ordinal_encoder.fit(working_train_df[ordinal_cols])
    if one_hot_cols:
        one_hot_encoder.fit(working_train_df[one_hot_cols])

    # Transform all the datasets
    working_train_df = transform_dataframe(
        working_train_df, ordinal_encoder, one_hot_encoder, ordinal_cols, one_hot_cols
    )
    working_val_df = transform_dataframe(
        working_val_df, ordinal_encoder, one_hot_encoder, ordinal_cols, one_hot_cols
    )
    working_test_df = transform_dataframe(
        working_test_df, ordinal_encoder, one_hot_encoder, ordinal_cols, one_hot_cols
    )

    # Initialize imputer
    imputer = SimpleImputer(strategy="median")

    # Fit the imputer only on the train data
    imputer.fit(working_train_df)

    # Transform all the datasets
    working_test_df = pd.DataFrame(
        imputer.transform(working_test_df), columns=working_test_df.columns
    )
    working_val_df = pd.DataFrame(
        imputer.transform(working_val_df), columns=working_val_df.columns
    )
    working_train_df = pd.DataFrame(
        imputer.transform(working_train_df), columns=working_train_df.columns
    )

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    # Create the scaler
    scaler = MinMaxScaler()

    # Fit the scaler only on the train data
    scaler.fit(working_train_df)

    # Use the fitted scaler to transform all the datasets
    working_train_df = pd.DataFrame(
        scaler.transform(working_train_df), columns=working_train_df.columns
    )
    working_val_df = pd.DataFrame(
        scaler.transform(working_val_df), columns=working_val_df.columns
    )
    working_test_df = pd.DataFrame(
        scaler.transform(working_test_df), columns=working_test_df.columns
    )

    return working_train_df.values, working_val_df.values, working_test_df.values


def transform_dataframe(
    df: pd.DataFrame,
    ordinal_encoder: OrdinalEncoder,
    one_hot_encoder: OneHotEncoder,
    ordinal_cols: list,
    one_hot_cols: list,
) -> pd.DataFrame:
    """
    Applies ordinal and one-hot encoding transformations to the given DataFrame.

    Parameters:
        working_df: pd.DataFrame
        ordinal_columns: list
        one_hot_columns: list
        ordinal_encoder: OrdinalEncoder
        onehot_encoder: OneHotEncoder

    Returns:
        pd.DataFrame
    """
    if ordinal_cols:
        df[ordinal_cols] = ordinal_encoder.transform(df[ordinal_cols])
    if one_hot_cols:
        # Get the one-hot transformed columns
        one_hot_transformed = one_hot_encoder.transform(df[one_hot_cols])
        # Get the column names
        one_hot_cols_names = one_hot_encoder.get_feature_names_out(one_hot_cols)
        # Create a dataframe with the one-hot columns with the same index as the original df
        one_hot_df = pd.DataFrame(
            one_hot_transformed, columns=one_hot_cols_names, index=df.index
        )
        # Drop the original columns and concatenate the one-hot columns
        df.drop(columns=one_hot_cols, inplace=True)
        return pd.concat([df, one_hot_df], axis=1)
    else:
        return df
