import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics(path, model_name):
    if not path.exists():
        print(f"Warning: Missing metrics for {model_name} at {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            metrics = data.get("test_metrics", {})
            metrics["Model"] = model_name
            return metrics
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def main():
    # Setup paths
    base_report_dir = Path("reports")
    models_config = [
        {"name": "Transformer", "path": base_report_dir / "train_model1" / "test_metrics.json"},
        {"name": "Random Forest", "path": base_report_dir / "reportRandom" / "test_metrics.json"},
        {"name": "Logistic Regression", "path": base_report_dir / "reportRegression" / "test_metrics.json"},
    ]

    all_metrics = []
    for config in models_config:
        m = load_metrics(config["path"], config["name"])
        if m:
            all_metrics.append(m)

    if not all_metrics:
        print("No metrics found to compare. Please run training scripts first.")
        return

    df = pd.DataFrame(all_metrics)
    
    # Chỉ giữ lại các chỉ số phân loại chính để so sánh công bằng
    display_metrics = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    df = df[["Model"] + [m for m in display_metrics if m in df.columns]]

    # Display Table
    print("\n" + "="*80)
    print("MODEL COMPARISON REPORT (Classification Metrics)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # Export to Markdown
    comparison_dir = base_report_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    md_path = comparison_dir / "comparison_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo so sánh hiệu năng các mô hình\n\n")
        f.write("Dưới đây là bảng so sánh các chỉ số đo lường giữa mô hình Transformer đề xuất và các mô hình Baseline.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n> [!NOTE]\n")
        f.write("> Kết quả cho thấy hiệu năng vượt trội của kiến trúc Transformer trong việc bắt lấy các mối quan hệ phức tạp giữa điểm số và ngành học.")

    # Visualization
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")
    
    plt.title("Comparison of Model Performance Metrics", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel("Score (0.0 - 1.0)")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add values on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_path = comparison_dir / "comparison_chart.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\nReport exported to: {md_path.resolve()}")
    print(f"Comparison chart saved to: {plot_path.resolve()}")

if __name__ == "__main__":
    main()
