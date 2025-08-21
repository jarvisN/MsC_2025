import pandas as pd
from visualizations import plot_price_with_ma, plot_monthly_avg, plot_volume_vs_price, plot_returns_distribution, plot_high_low_range
from analysis import load_data, compute_summary

def main():
    # Load dataset
    df = load_data("data/gold_price_data.csv")

    # Summary
    summary = compute_summary(df)
    print("Dataset Summary:", summary)

    # Visualizations
    plot_price_with_ma(df)
    plot_monthly_avg(df)
    plot_volume_vs_price(df)
    plot_returns_distribution(df)
    plot_high_low_range(df)

if __name__ == "__main__":
    main()
