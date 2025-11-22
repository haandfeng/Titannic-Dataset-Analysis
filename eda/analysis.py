"""探索性数据分析模块"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
from config import Config


class EDAAnalyzer:
    """探索性数据分析器"""
    
    def __init__(self, plot_enabled: bool = None, save_plots: bool = False, 
                 plots_dir: Optional[Path] = None):
        """
        初始化EDA分析器
        
        Args:
            plot_enabled: 是否显示图表
            save_plots: 是否保存图表
            plots_dir: 图表保存目录
        """
        self.plot_enabled = plot_enabled if plot_enabled is not None else Config.EDA_PLOT_ENABLED
        self.save_plots = save_plots
        self.plots_dir = plots_dir or Config.PLOTS_DIR
        if self.save_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def overall_survival_rate(self, df: pd.DataFrame, plot: bool = None) -> pd.DataFrame:
        """计算整体生存率统计"""
        plot = plot if plot is not None else self.plot_enabled
        
        counts = df["Survived"].value_counts().sort_index()
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
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(summary["Survived"], summary["Rate"])
            ax.set_ylabel("Survival Rate")
            ax.set_title("Overall Survival Rate")
            ax.set_ylim(0, 1)
            if self.save_plots:
                plt.savefig(self.plots_dir / "overall_survival_rate.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        return summary
    
    def survival_by_sex(self, df: pd.DataFrame, plot: bool = None) -> pd.DataFrame:
        """按性别分析生存率"""
        plot = plot if plot is not None else self.plot_enabled
        
        grouped = df.groupby("Sex")["Survived"]
        summary = grouped.agg(
            Count="count",
            Survived_Sum="sum",
            Survival_Rate="mean"
        ).reset_index()
        
        print("=== Survival by Sex ===")
        print(summary)
        
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(summary["Sex"], summary["Survival_Rate"])
            ax.set_ylabel("Survival Rate")
            ax.set_title("Survival Rate by Sex")
            ax.set_ylim(0, 1)
            if self.save_plots:
                plt.savefig(self.plots_dir / "survival_by_sex.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        return summary
    
    def survival_by_pclass(self, df: pd.DataFrame, plot: bool = None) -> pd.DataFrame:
        """按舱位等级分析生存率"""
        plot = plot if plot is not None else self.plot_enabled
        
        grouped = df.groupby("Pclass")["Survived"]
        summary = grouped.agg(
            Count="count",
            Survived_Sum="sum",
            Survival_Rate="mean"
        ).reset_index().sort_values("Pclass")
        
        print("=== Survival by Passenger Class (Pclass) ===")
        print(summary)
        
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(summary["Pclass"].astype(str), summary["Survival_Rate"])
            ax.set_xlabel("Pclass")
            ax.set_ylabel("Survival Rate")
            ax.set_title("Survival Rate by Passenger Class")
            ax.set_ylim(0, 1)
            if self.save_plots:
                plt.savefig(self.plots_dir / "survival_by_pclass.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        return summary
    
    def age_distribution_and_survival(self, df: pd.DataFrame, bins=None, 
                                     plot: bool = None) -> pd.DataFrame:
        """分析年龄分布和生存率"""
        plot = plot if plot is not None else self.plot_enabled
        
        print("=== Age Summary (ignoring missing Age) ===")
        age_desc = df["Age"].describe()
        print(age_desc)
        
        if bins is None:
            bins = [0, 10, 20, 30, 40, 50, 60, 80]
        
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
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(sub["Age"], bins=20)
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            ax.set_title("Age Distribution")
            if self.save_plots:
                plt.savefig(self.plots_dir / "age_distribution.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x_labels = summary["AgeBin"].astype(str)
            ax.bar(x_labels, summary["Survival_Rate"])
            ax.set_ylabel("Survival Rate")
            ax.set_title("Survival Rate by Age Bin")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if self.save_plots:
                plt.savefig(self.plots_dir / "survival_by_age_bin.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        return summary
    
    def family_size_and_isalone(self, df: pd.DataFrame, plot: bool = None) -> Dict:
        """分析家庭规模和是否独自出行"""
        plot = plot if plot is not None else self.plot_enabled
        
        df = df.copy()
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        
        print("=== Basic FamilySize & IsAlone Summary ===")
        print(df[["FamilySize", "IsAlone"]].describe())
        
        fs_grouped = df.groupby("FamilySize")["Survived"].agg(
            Count="count",
            Survived_Sum="sum",
            Survival_Rate="mean"
        ).reset_index()
        
        print("\n=== Survival by FamilySize ===")
        print(fs_grouped)
        
        ia_grouped = df.groupby("IsAlone")["Survived"].agg(
            Count="count",
            Survived_Sum="sum",
            Survival_Rate="mean"
        ).reset_index()
        
        print("\n=== Survival by IsAlone (0 = has family, 1 = alone) ===")
        print(ia_grouped)
        
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(fs_grouped["FamilySize"].astype(str), fs_grouped["Survival_Rate"])
            ax.set_xlabel("FamilySize")
            ax.set_ylabel("Survival Rate")
            ax.set_title("Survival Rate by FamilySize")
            ax.set_ylim(0, 1)
            if self.save_plots:
                plt.savefig(self.plots_dir / "survival_by_family_size.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            x_labels = ["Has family (0)", "Alone (1)"]
            ax.bar(x_labels, ia_grouped["Survival_Rate"])
            ax.set_ylabel("Survival Rate")
            ax.set_title("Survival Rate by IsAlone")
            ax.set_ylim(0, 1)
            if self.save_plots:
                plt.savefig(self.plots_dir / "survival_by_isalone.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        return {
            "family_size_summary": fs_grouped,
            "isalone_summary": ia_grouped
        }
    
    def embarked_and_survival(self, df: pd.DataFrame, plot: bool = None) -> pd.DataFrame:
        """分析登船港口与生存率"""
        plot = plot if plot is not None else self.plot_enabled
        
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
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(grouped["Embarked"], grouped["Survival_Rate"])
            ax.set_xlabel("Embarked")
            ax.set_ylabel("Survival Rate")
            ax.set_title("Survival Rate by Embarked Port")
            ax.set_ylim(0, 1)
            if self.save_plots:
                plt.savefig(self.plots_dir / "survival_by_embarked.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        return grouped
    
    def fare_and_cabin(self, df: pd.DataFrame, fare_bins=None, 
                      plot: bool = None) -> Dict:
        """分析票价和船舱信息"""
        plot = plot if plot is not None else self.plot_enabled
        
        df = df.copy()
        
        print("=== Fare Summary (ignoring missing Fare) ===")
        fare_desc = df["Fare"].describe()
        print(fare_desc)
        
        if fare_bins is None:
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
        
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        cabin_grouped = df.groupby("HasCabin")["Survived"].agg(
            Count="count",
            Survived_Sum="sum",
            Survival_Rate="mean"
        ).reset_index()
        
        print("\n=== Survival by HasCabin (0 = no cabin info, 1 = has cabin info) ===")
        print(cabin_grouped)
        
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df["Fare"], bins=30)
            ax.set_xlabel("Fare")
            ax.set_ylabel("Count")
            ax.set_title("Fare Distribution")
            if self.save_plots:
                plt.savefig(self.plots_dir / "fare_distribution.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x_labels = fare_grouped["FareBin"].astype(str)
            ax.bar(x_labels, fare_grouped["Survival_Rate"])
            ax.set_ylabel("Survival Rate")
            ax.set_title("Survival Rate by Fare Bin")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if self.save_plots:
                plt.savefig(self.plots_dir / "survival_by_fare_bin.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            x_labels = ["No cabin (0)", "Has cabin (1)"]
            ax.bar(x_labels, cabin_grouped["Survival_Rate"])
            ax.set_ylabel("Survival Rate")
            ax.set_title("Survival Rate by Cabin Info")
            ax.set_ylim(0, 1)
            if self.save_plots:
                plt.savefig(self.plots_dir / "survival_by_cabin.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        return {
            "fare_bins_summary": fare_grouped,
            "cabin_missing_summary": cabin_grouped
        }
    
    def run_full_analysis(self, df: pd.DataFrame):
        """运行完整的EDA分析"""
        print("=" * 60)
        print("开始完整的探索性数据分析")
        print("=" * 60)
        
        self.overall_survival_rate(df)
        self.survival_by_sex(df)
        self.survival_by_pclass(df)
        self.age_distribution_and_survival(df)
        self.family_size_and_isalone(df)
        self.embarked_and_survival(df)
        self.fare_and_cabin(df)
        
        print("\n" + "=" * 60)
        print("EDA分析完成")
        print("=" * 60)

