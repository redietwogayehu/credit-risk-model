import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =========================================================
# 1. DATA LOADING
# =========================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load raw transaction dataset.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


# =========================================================
# 2. DATA QUALITY CHECKS
# =========================================================

def data_overview(df: pd.DataFrame):
    """
    Print basic dataset overview.
    """
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nInfo:")
    print(df.info())


def missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Return missing value counts per column.
    """
    return df.isnull().sum()


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for numerical features.
    """
    return df.describe()


# =========================================================
# 3. VISUALIZATION FUNCTIONS (EDA)
# =========================================================

def plot_missing_heatmap(df: pd.DataFrame):
    """
    Visualize missing values.
    """
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Values Heatmap")
    plt.show()


def plot_histogram(df: pd.DataFrame, column: str):
    """
    Plot distribution of numerical feature.
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_boxplot(df: pd.DataFrame, column: str):
    """
    Detect outliers using boxplot.
    """
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()


def plot_countplot(df: pd.DataFrame, column: str):
    """
    Categorical distribution plot.
    """
    plt.figure(figsize=(8, 4))
    sns.countplot(x=column, data=df)
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_correlation(df: pd.DataFrame, cols: list):
    """
    Correlation heatmap for selected numerical columns.
    """
    plt.figure(figsize=(6, 4))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


# =========================================================
# 4. FEATURE ENGINEERING (TIME FEATURES)
# =========================================================

def extract_datetime_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Extract time-based features from datetime column.
    """
    df[column] = pd.to_datetime(df[column])

    df["Year"] = df[column].dt.year
    df["Month"] = df[column].dt.month
    df["Day"] = df[column].dt.day
    df["Hour"] = df[column].dt.hour

    return df


# =========================================================
# 5. BASIC ANALYSIS WRAPPER 
# =========================================================

def run_basic_eda(df: pd.DataFrame):
    """
    Run quick standardized EDA pipeline.
    """
    data_overview(df)

    print("\nMissing Values:\n", missing_values(df))
    print("\nSummary Statistics:\n", describe_data(df))

    plot_missing_heatmap(df)
    plot_histogram(df, "Amount")
    plot_histogram(df, "Value")
    plot_boxplot(df, "Amount")
    plot_countplot(df, "ChannelId")
    plot_correlation(df, ["Amount", "Value"])