import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_train(path="train.csv"):
    """Load the Titanic training set."""
    return pd.read_csv(path)

def eda_overall_survival_rate(df: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """
    Compute overall survival counts and rates, and optionally plot a bar chart.
    """
    counts = df["Survived"].value_counts().sort_index()  # 0, 1
    total = len(df)
    rates = counts / total

    summary = pd.DataFrame({
        "Survived": ["No (0)", "Yes (1)"],
        "Count": counts.values,
        "Rate": rates.values
    })
    print("=== Overall Survival Rate ===")
    print(summary)
    print(f"\nTotal passengers: {total}")

    if plot:
        fig, ax = plt.subplots()
        ax.bar(summary["Survived"], summary["Rate"])
        ax.set_ylabel("Survival Rate")
        ax.set_title("Overall Survival Rate")
        ax.set_ylim(0, 1)
        plt.show()

    return summary

def eda_survival_by_sex(df: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """
    Compute survival rate grouped by Sex, and optionally plot a bar chart.
    """
    grouped = df.groupby("Sex")["Survived"]
    summary = grouped.agg(
        Count="count",
        Survived_Sum="sum",
        Survival_Rate="mean"
    ).reset_index()

    print("=== Survival by Sex ===")
    print(summary)

    if plot:
        fig, ax = plt.subplots()
        ax.bar(summary["Sex"], summary["Survival_Rate"])
        ax.set_ylabel("Survival Rate")
        ax.set_title("Survival Rate by Sex")
        ax.set_ylim(0, 1)
        plt.show()

    return summary

def eda_survival_by_pclass(df: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """
    Compute survival rate grouped by passenger class (Pclass),
    and optionally plot a bar chart.
    """
    grouped = df.groupby("Pclass")["Survived"]
    summary = grouped.agg(
        Count="count",
        Survived_Sum="sum",
        Survival_Rate="mean"
    ).reset_index().sort_values("Pclass")

    print("=== Survival by Passenger Class (Pclass) ===")
    print(summary)

    if plot:
        fig, ax = plt.subplots()
        ax.bar(summary["Pclass"].astype(str), summary["Survival_Rate"])
        ax.set_xlabel("Pclass")
        ax.set_ylabel("Survival Rate")
        ax.set_title("Survival Rate by Passenger Class")
        ax.set_ylim(0, 1)
        plt.show()

    return summary

def eda_age_distribution_and_survival(df: pd.DataFrame,
                                      bins=None,
                                      plot: bool = True) -> pd.DataFrame:
    """
    Analyze age distribution and survival by age bins.
    Optionally plot histograms / bar chart.
    """
    print("=== Age Summary (ignoring missing Age) ===")
    age_desc = df["Age"].describe()
    print(age_desc)

    # Define age bins
    if bins is None:
        bins = [0, 10, 20, 30, 40, 50, 60, 80]

    # Drop missing ages for binning
    sub = df.dropna(subset=["Age"]).copy()
    sub["AgeBin"] = pd.cut(sub["Age"], bins=bins, right=False, include_lowest=True)

    grouped = sub.groupby("AgeBin")["Survived"]
    summary = grouped.agg(
        Count="count",
        Survived_Sum="sum",
        Survival_Rate="mean"
    ).reset_index()

    print("\n=== Survival by Age Bins ===")
    print(summary)

    if plot:
        # 1) Age histogram
        fig, ax = plt.subplots()
        ax.hist(sub["Age"], bins=20)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.set_title("Age Distribution")
        plt.show()

        # 2) Survival rate by age bin
        fig, ax = plt.subplots()
        x_labels = summary["AgeBin"].astype(str)
        ax.bar(x_labels, summary["Survival_Rate"])
        ax.set_ylabel("Survival Rate")
        ax.set_title("Survival Rate by Age Bin")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return summary

def eda_family_size_and_isalone(df: pd.DataFrame, plot: bool = True):
    """
    Create FamilySize and IsAlone features, then compute survival rate by these,
    and optionally plot bar charts.
    """
    df = df.copy()
    # Create FamilySize
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    # Create IsAlone
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    print("=== Basic FamilySize & IsAlone Summary ===")
    print(df[["FamilySize", "IsAlone"]].describe())

    # Survival by FamilySize
    fs_grouped = df.groupby("FamilySize")["Survived"].agg(
        Count="count",
        Survived_Sum="sum",
        Survival_Rate="mean"
    ).reset_index()

    print("\n=== Survival by FamilySize ===")
    print(fs_grouped)

    # Survival by IsAlone
    ia_grouped = df.groupby("IsAlone")["Survived"].agg(
        Count="count",
        Survived_Sum="sum",
        Survival_Rate="mean"
    ).reset_index()

    print("\n=== Survival by IsAlone (0 = has family, 1 = alone) ===")
    print(ia_grouped)

    if plot:
        # FamilySize vs Survival rate
        fig, ax = plt.subplots()
        ax.bar(fs_grouped["FamilySize"].astype(str), fs_grouped["Survival_Rate"])
        ax.set_xlabel("FamilySize")
        ax.set_ylabel("Survival Rate")
        ax.set_title("Survival Rate by FamilySize")
        ax.set_ylim(0, 1)
        plt.show()

        # IsAlone vs Survival rate
        fig, ax = plt.subplots()
        x_labels = ["Has family (0)", "Alone (1)"]
        ax.bar(x_labels, ia_grouped["Survival_Rate"])
        ax.set_ylabel("Survival Rate")
        ax.set_title("Survival Rate by IsAlone")
        ax.set_ylim(0, 1)
        plt.show()

    return {
        "family_size_summary": fs_grouped,
        "isalone_summary": ia_grouped
    }

def eda_embarked_and_survival(df: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """
    Analyze survival rates by Embarked port, and optionally plot a bar chart.
    """
    sub = df.copy()
    sub["Embarked"] = sub["Embarked"].fillna("Unknown")

    grouped = sub.groupby("Embarked")["Survived"].agg(
        Count="count",
        Survived_Sum="sum",
        Survival_Rate="mean"
    ).reset_index()

    print("=== Survival by Embarked Port ===")
    print(grouped)

    if plot:
        fig, ax = plt.subplots()
        ax.bar(grouped["Embarked"], grouped["Survival_Rate"])
        ax.set_xlabel("Embarked")
        ax.set_ylabel("Survival Rate")
        ax.set_title("Survival Rate by Embarked Port")
        ax.set_ylim(0, 1)
        plt.show()

    return grouped

def eda_fare_and_cabin(df: pd.DataFrame,
                       fare_bins=None,
                       plot: bool = True) -> dict:
    """
    Analyze Fare (ticket price) and Cabin information in relation to survival.
    Optionally plot histograms and bar charts.
    """
    df = df.copy()

    # === Fare Analysis ===
    print("=== Fare Summary (ignoring missing Fare) ===")
    fare_desc = df["Fare"].describe()
    print(fare_desc)

    if fare_bins is None:
        # 简单的分位数分箱：四分位
        fare_bins = [-0.01,
                     df["Fare"].quantile(0.25),
                     df["Fare"].quantile(0.5),
                     df["Fare"].quantile(0.75),
                     df["Fare"].max() + 1]

    df["FareBin"] = pd.cut(df["Fare"], bins=fare_bins, right=False, include_lowest=True)

    fare_grouped = df.groupby("FareBin")["Survived"].agg(
        Count="count",
        Survived_Sum="sum",
        Survival_Rate="mean"
    ).reset_index()

    print("\n=== Survival by Fare Bins ===")
    print(fare_grouped)

    # === Cabin Analysis ===
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    cabin_grouped = df.groupby("HasCabin")["Survived"].agg(
        Count="count",
        Survived_Sum="sum",
        Survival_Rate="mean"
    ).reset_index()

    print("\n=== Survival by HasCabin (0 = no cabin info, 1 = has cabin info) ===")
    print(cabin_grouped)

    if plot:
        # Fare histogram
        fig, ax = plt.subplots()
        ax.hist(df["Fare"], bins=30)
        ax.set_xlabel("Fare")
        ax.set_ylabel("Count")
        ax.set_title("Fare Distribution")
        plt.show()

        # Survival rate by fare bin
        fig, ax = plt.subplots()
        x_labels = fare_grouped["FareBin"].astype(str)
        ax.bar(x_labels, fare_grouped["Survival_Rate"])
        ax.set_ylabel("Survival Rate")
        ax.set_title("Survival Rate by Fare Bin")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Survival rate by HasCabin
        fig, ax = plt.subplots()
        x_labels = ["No cabin (0)", "Has cabin (1)"]
        ax.bar(x_labels, cabin_grouped["Survival_Rate"])
        ax.set_ylabel("Survival Rate")
        ax.set_title("Survival Rate by Cabin Info")
        ax.set_ylim(0, 1)
        plt.show()

    return {
        "fare_bins_summary": fare_grouped,
        "cabin_missing_summary": cabin_grouped
    }

if __name__ == "__main__":
    df = load_train(r"C:\Users\zhy20\Desktop\研一\ECE225\project\titanic\dataset\train.csv")

    overall = eda_overall_survival_rate(df, plot=True)
    by_sex = eda_survival_by_sex(df, plot=True)
    by_pclass = eda_survival_by_pclass(df, plot=True)
    age_summary = eda_age_distribution_and_survival(df, plot=True)
    family_results = eda_family_size_and_isalone(df, plot=True)
    embarked_summary = eda_embarked_and_survival(df, plot=True)
    fare_cabin_results = eda_fare_and_cabin(df, plot=True)