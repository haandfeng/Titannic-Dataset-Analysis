"""逻辑回归特征影响分析脚本"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data import DataLoader, DataPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from config import Config

# 设置样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 统一颜色
UNIFIED_COLOR = 'slategray'

def train_logistic_model():
    """训练逻辑回归模型并提取系数"""
    print("="*60)
    print("训练逻辑回归模型（包含FamilySize和IsAlone特征）")
    print("="*60)
    
    # 加载数据
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.load_train_test_split()
    
    # 预处理（包含FamilySize和IsAlone）
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 训练模型
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_processed, y_train)
    
    # 评估
    y_pred = model.predict(X_test_processed)
    print(f"\n模型性能:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    
    # 获取特征名称
    feature_names = []
    if len(preprocessor.numeric_features) > 0:
        feature_names.extend(preprocessor.numeric_features)
    if len(preprocessor.categorical_features) > 0:
        encoder = preprocessor.onehot_encoder
        cat_feature_names = encoder.get_feature_names_out(preprocessor.categorical_features).tolist()
        feature_names.extend(cat_feature_names)
    
    # 获取系数
    coef = model.coef_[0]
    coeff_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
    })
    coeff_df['abs_coefficient'] = coeff_df['coefficient'].abs()
    coeff_df['direction'] = coeff_df['coefficient'].apply(
        lambda c: 'Positive (增加生存概率)' if c > 0 else 'Negative (降低生存概率)'
    )
    coeff_df = coeff_df.sort_values('abs_coefficient', ascending=False)
    
    return coeff_df, preprocessor

def categorize_feature(feature):
    """分类特征"""
    if feature.startswith('Sex_'):
        return 'Sex'
    elif feature.startswith('Pclass_'):
        return 'Pclass'
    elif feature.startswith('Embarked_'):
        return 'Embarked'
    elif feature.startswith('Ticket_'):
        return 'Ticket'
    elif feature.startswith('Cabin_'):
        return 'Cabin'
    elif feature.startswith('Name_'):
        return 'Name'
    elif feature in ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']:
        return feature
    else:
        return 'Other'

def create_visualizations(coeff_df):
    """创建可视化图表"""
    Config.create_directories()
    plots_dir = Config.PLOTS_DIR
    
    # 添加特征类别
    coeff_df['category'] = coeff_df['feature'].apply(categorize_feature)
    
    # 提取关键特征
    key_categories = ['Sex', 'Pclass', 'Embarked', 'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
    key_features_df = coeff_df[coeff_df['category'].isin(key_categories)].copy()
    
    # 保存关键特征（按系数绝对值排序）
    key_features_df_sorted = key_features_df.sort_values('abs_coefficient', ascending=False)
    key_features_df_sorted.to_csv(Config.OUTPUT_DIR / 'key_features_coefficients.csv', index=False)
    
    # 1. 条形图 - 按特征名称字母顺序排序
    key_features_df_alphabetical = key_features_df.sort_values('feature', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in key_features_df_alphabetical['coefficient']]
    bars = ax.barh(range(len(key_features_df_alphabetical)), key_features_df_alphabetical['coefficient'], 
                   color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(range(len(key_features_df_alphabetical)))
    ax.set_yticklabels(key_features_df_alphabetical['feature'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Logistic Regression Feature Coefficients\n(Sorted Alphabetically by Feature Name)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加数值标签
    for i, (bar, coef) in enumerate(zip(bars, key_features_df_alphabetical['coefficient'])):
        ax.text(coef + (0.02 if coef > 0 else -0.02), i, 
               f'{coef:.3f}', 
               va='center', ha='left' if coef > 0 else 'right',
               fontsize=10, fontweight='bold')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', alpha=0.8, label='Positive (Increases Survival)'),
        Patch(facecolor='#FF6B6B', alpha=0.8, label='Negative (Decreases Survival)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    bar_path = plots_dir / 'logistic_coefficients_barplot.png'
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"\n保存条形图: {bar_path}")
    plt.close()
    
    # 2. 热力图 - 按特征类别分组
    # 准备热力图数据
    heatmap_data = []
    for category in key_categories:
        cat_features = key_features_df[key_features_df['category'] == category]
        if len(cat_features) > 0:
            for _, row in cat_features.iterrows():
                heatmap_data.append({
                    'Category': category,
                    'Feature': row['feature'],
                    'Coefficient': row['coefficient']
                })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # 创建透视表用于热力图
    pivot_data = heatmap_df.pivot_table(
        values='Coefficient', 
        index='Feature', 
        columns='Category',
        fill_value=0
    )
    
    # 如果只有一个类别，转置
    if pivot_data.shape[1] == 1:
        pivot_data = pivot_data.T
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot_data) * 0.4)))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Coefficient'},
                linewidths=0.5, linecolor='white',
                square=False, ax=ax)
    
    ax.set_title('Logistic Regression Feature Coefficients Heatmap\n(Grouped by Feature Category)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Feature Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    heatmap_path = plots_dir / 'logistic_coefficients_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"保存热力图: {heatmap_path}")
    plt.close()
    
    # 3. 按类别汇总的条形图
    category_summary = key_features_df.groupby('category').agg({
        'abs_coefficient': 'mean'
    }).sort_values('abs_coefficient', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(category_summary)), category_summary['abs_coefficient'],
                   color=UNIFIED_COLOR, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(range(len(category_summary)))
    ax.set_yticklabels(category_summary.index, fontsize=11, fontweight='bold')
    ax.set_xlabel('Average Absolute Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Average Feature Impact by Category\n(Higher Value = Greater Impact)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, category_summary['abs_coefficient'])):
        ax.text(val + 0.01, i, f'{val:.3f}', 
               va='center', ha='left', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    category_path = plots_dir / 'logistic_coefficients_by_category.png'
    plt.savefig(category_path, dpi=300, bbox_inches='tight')
    print(f"保存类别汇总图: {category_path}")
    plt.close()
    
    return key_features_df

def generate_analysis_report(key_features_df):
    """生成分析报告"""
    print("\n" + "="*60)
    print("特征影响分析报告")
    print("="*60)
    
    print("\n【Top 10 最重要特征（按系数绝对值排序）】")
    print(key_features_df.head(10)[['feature', 'coefficient', 'abs_coefficient', 'direction']].to_string(index=False))
    
    print("\n【按特征类别汇总】")
    category_summary = key_features_df.groupby('category').agg({
        'coefficient': ['mean', 'count'],
        'abs_coefficient': 'mean'
    })
    category_summary.columns = ['Mean_Coefficient', 'Count', 'Mean_Abs_Coefficient']
    category_summary = category_summary.sort_values('Mean_Abs_Coefficient', ascending=False)
    print(category_summary.to_string())
    
    # 保存报告
    report_path = Config.OUTPUT_DIR / 'logistic_feature_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("逻辑回归特征影响分析报告\n")
        f.write("="*60 + "\n\n")
        f.write("【Top 10 最重要特征（按系数绝对值排序）】\n")
        f.write(key_features_df.head(10)[['feature', 'coefficient', 'abs_coefficient', 'direction']].to_string(index=False))
        f.write("\n\n【按特征类别汇总】\n")
        f.write(category_summary.to_string())
        f.write("\n\n【详细特征系数】\n")
        f.write(key_features_df[['feature', 'coefficient', 'abs_coefficient', 'direction']].to_string(index=False))
    
    print(f"\n保存分析报告: {report_path}")

if __name__ == "__main__":
    # 训练模型并提取系数
    coeff_df, preprocessor = train_logistic_model()
    
    # 保存完整系数
    coeff_df.to_csv(Config.OUTPUT_DIR / 'logistic_coefficients_with_features.csv', index=False)
    print(f"\n保存完整系数文件: {Config.OUTPUT_DIR / 'logistic_coefficients_with_features.csv'}")
    print(f"总特征数量: {len(coeff_df)}")
    
    # 创建可视化
    key_features_df = create_visualizations(coeff_df)
    
    # 生成分析报告
    generate_analysis_report(key_features_df)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)

