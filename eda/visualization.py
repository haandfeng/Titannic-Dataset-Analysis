"""数据可视化模块 - 只包含boxplots和key_features_distribution"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from config import Config

# Set English font and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Unified color: slategray (灰蓝色)
UNIFIED_COLOR = 'slategray'


class DataVisualizer:
    """数据可视化器 - 只生成boxplots和key_features_distribution"""
    
    def __init__(self, save_plots: bool = True, plots_dir: Optional[Path] = None):
        """
        Initialize visualizer
        
        Args:
            save_plots: Whether to save plots
            plots_dir: Directory to save plots
        """
        self.save_plots = save_plots
        self.plots_dir = plots_dir or Config.PLOTS_DIR
        if self.save_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features: FamilySize, IsAlone
        
        Args:
            df: Original dataframe
            
        Returns:
            Dataframe with prepared features
        """
        df_work = df.copy()
        
        # Create FamilySize
        df_work['FamilySize'] = df_work['SibSp'] + df_work['Parch'] + 1
        
        # Create IsAlone
        df_work['IsAlone'] = (df_work['FamilySize'] == 1).astype(int)
        
        return df_work
    
    def plot_key_features_distribution(self, df: pd.DataFrame, figsize=(16, 12)):
        """
        Plot key features distribution: FamilySize, IsAlone, Sex, Age, Pclass, Fare
        
        Args:
            df: Dataframe with original data
            figsize: Figure size
        """
        # Prepare features
        df_work = self.prepare_features(df)
        
        # Key features from analysis.py
        key_features = {
            'FamilySize': 'numeric',
            'IsAlone': 'numeric',
            'Sex': 'categorical',
            'Age': 'numeric',
            'Pclass': 'numeric',
            'Fare': 'numeric'
        }
        
        # Create subplots: 3 rows x 2 columns
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (feature, ftype) in enumerate(key_features.items()):
            ax = axes[idx]
            
            if ftype == 'numeric':
                data = df_work[feature].dropna()
                
                if len(data) > 0:
                    # Histogram with unified color
                    n, bins, patches = ax.hist(data, bins=30, color=UNIFIED_COLOR, 
                                              edgecolor='white', alpha=0.8, linewidth=0.5)
                    
                    # Add statistics
                    mean_val = data.mean()
                    median_val = data.median()
                    std_val = data.std()
                    
                    ax.axvline(mean_val, color='#FF6B6B', linestyle='--', linewidth=2.5, 
                              label=f'Mean: {mean_val:.2f}', zorder=3)
                    ax.axvline(median_val, color='#4ECDC4', linestyle='--', linewidth=2.5, 
                              label=f'Median: {median_val:.2f}', zorder=3)
                    
                    ax.set_title(f'{feature}\n(Mean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f})', 
                               fontsize=12, fontweight='bold', pad=10)
                    ax.set_xlabel('Value', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                    ax.legend(fontsize=9, framealpha=0.9)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                else:
                    ax.text(0.5, 0.5, f'No data for {feature}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{feature} (No Data)', fontsize=12, fontweight='bold')
            
            elif ftype == 'categorical':
                value_counts = df_work[feature].value_counts()
                
                if len(value_counts) > 0:
                    # Bar chart with unified color
                    bars = ax.bar(range(len(value_counts)), value_counts.values, 
                                color=UNIFIED_COLOR, edgecolor='white', alpha=0.8, linewidth=1.5)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, fontsize=10, fontweight='bold')
                    ax.set_title(f'{feature}\n(Unique: {df_work[feature].nunique()})', 
                               fontsize=12, fontweight='bold', pad=10)
                    ax.set_xlabel('Category', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Add count labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}\n({height/len(df_work)*100:.1f}%)',
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'No data for {feature}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{feature} (No Data)', fontsize=12, fontweight='bold')
        
        plt.suptitle('Key Features Distribution', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Always save the plot
        save_path = self.plots_dir / "key_features_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if self.save_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_boxplots(self, df: pd.DataFrame, target_col: str = 'Survived', figsize=(15, 10)):
        """
        Plot boxplots comparing numeric features by survival status
        
        Args:
            df: Dataframe with original data
            target_col: Target column name
            figsize: Figure size
        """
        if target_col not in df.columns:
            print(f"Target column {target_col} does not exist")
            return
        
        # Prepare features
        df_work = self.prepare_features(df)
        
        # Get numeric columns from analysis.py exploration (exclude target and ID columns)
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [target_col, 'PassengerId', 'Survived']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) == 0:
            print("No numeric features to plot")
            return
        
        # Calculate subplot layout
        cols_per_row = 3
        rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=figsize)
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes] if rows == 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            # Prepare data for each target value
            data_to_plot = []
            labels = []
            target_values = sorted(df_work[target_col].unique())
            
            for target_val in target_values:
                data = df_work[df_work[target_col] == target_val][col].dropna()
                if len(data) > 0:
                    data_to_plot.append(data)
                    # Better labels
                    if target_col == 'Survived':
                        labels.append('Not Survived' if target_val == 0 else 'Survived')
                    else:
                        labels.append(f'{target_col}={target_val}')
            
            if len(data_to_plot) > 0:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                               widths=0.6, showmeans=True, meanline=True)
                
                # Style boxplots with unified color
                for patch in bp['boxes']:
                    patch.set_facecolor(UNIFIED_COLOR)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('white')
                    patch.set_linewidth(1.5)
                
                # Style other elements
                for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                    for item in bp[element]:
                        item.set_color(UNIFIED_COLOR)
                        item.set_linewidth(1.5)
                
                # Style medians
                for median in bp['medians']:
                    median.set_color('#FF6B6B')
                    median.set_linewidth(2.5)
                
                # Style means
                for mean in bp['means']:
                    mean.set_color('#4ECDC4')
                    mean.set_linewidth(2)
                    mean.set_linestyle('--')
                
                ax.set_title(col, fontsize=13, fontweight='bold', pad=10)
                ax.set_ylabel('Value', fontsize=11, fontweight='bold')
                ax.set_xlabel('', fontsize=10)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='x', labelsize=10, rotation=0)
                ax.tick_params(axis='y', labelsize=9)
        
        # Hide extra subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        # Add legend to the entire figure (bottom right corner)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#FF6B6B', linewidth=2.5, label='Median'),
            Line2D([0], [0], color='#4ECDC4', linewidth=2, linestyle='--', label='Mean')
        ]
        fig.legend(handles=legend_elements, loc='lower right', fontsize=10, 
                  framealpha=0.9, bbox_to_anchor=(0.98, 0.02))
        
        plt.suptitle('Numeric Features Distribution by Survival Status', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Always save the plot
        save_path = self.plots_dir / "boxplots.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if self.save_plots:
            plt.show()
        else:
            plt.close()
    
    def generate_visualizations(self, df: pd.DataFrame, target_col: str = 'Survived'):
        """
        Generate both visualizations
        
        Args:
            df: Original dataframe
            target_col: Target column name
        """
        print("\n" + "=" * 100)
        print("Generating Data Visualizations")
        print("=" * 100)
        
        # 1. Key features distribution
        print("\n1. Generating key features distribution...")
        self.plot_key_features_distribution(df)
        
        # 2. Boxplots
        if target_col in df.columns:
            print("\n2. Generating boxplots...")
            self.plot_boxplots(df, target_col)
        
        print("\n" + "=" * 100)
        print("Visualization completed!")
        print(f"All outputs saved to: {self.plots_dir}")
        print("=" * 100 + "\n")

