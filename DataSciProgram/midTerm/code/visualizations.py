import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_price_with_ma(df):
    plt.figure()
    df['adj_close'].plot(linewidth=1, label='Adj Close')
    df['adj_close'].rolling(window=50).mean().plot(linewidth=1, label='MA50')
    df['adj_close'].rolling(window=200).mean().plot(linewidth=1, label='MA200')
    plt.title('Gold Price with 50/200-Day Moving Averages')
    plt.xlabel('Date'); plt.ylabel('Price')
    plt.legend(); plt.tight_layout()
    plt.show()

def plot_monthly_avg(df):
    monthly_mean = df['adj_close'].resample('M').mean()
    plt.figure()
    monthly_mean.plot(linewidth=1.2)
    plt.title('Monthly Average Gold Price')
    plt.xlabel('Month'); plt.ylabel('Average Price')
    plt.tight_layout(); plt.show()

def plot_volume_vs_price(df):
    plt.figure()
    plt.scatter(df['adj_close'], df['volume'], s=5, alpha=0.5)
    plt.title('Volume vs. Adjusted Close')
    plt.xlabel('Adjusted Close'); plt.ylabel('Volume')
    plt.tight_layout(); plt.show()

def plot_returns_distribution(df):
    returns = df['adj_close'].pct_change().dropna()
    plt.figure()
    plt.hist(returns, bins=60)
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return'); plt.ylabel('Frequency')
    mu, sigma = returns.mean(), returns.std()
    print(f"Daily returns mean: {mu:.6f}, std: {sigma:.6f}")
    plt.tight_layout(); plt.show()

def plot_high_low_range(df):
    date_num = mdates.date2num(df.index.to_pydatetime())
    plt.figure()
    plt.fill_between(date_num, df['low'].values, df['high'].values, alpha=0.2, label='Low–High Range')
    plt.plot(date_num, df['close'].values, linewidth=0.8, label='Close')
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    plt.title('Daily High–Low Range and Close')
    plt.xlabel('Date'); plt.ylabel('Price')
    plt.legend(); plt.tight_layout()
    plt.show()
