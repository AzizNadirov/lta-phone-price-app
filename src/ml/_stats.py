import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import warnings
from loguru import logger
warnings.filterwarnings('ignore')



def plot_price_distributions(price_data: dict, 
                             figsize=(16, 12), 
                             save_path=None):
    """
    Analyze and plot price distributions for different models.
    
    Parameters:
    -----------
    price_data : dict
        Dictionary with format {'model_name': [price_1, price_2, ...]}
    figsize : tuple, optional
        Figure size for the plots (default: (16, 12))
    save_path : str, optional
        Path to save the plot (default: None, just display)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    
    df_list = []
    for model, prices in price_data.items():
        df_list.extend([{'model': model, 'price': price} for price in prices])
    
    df = pd.DataFrame(df_list)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Price Distribution Analysis Across Models', fontsize=16, fontweight='bold')
    
    # 1. Histogram with KDE overlay
    ax1 = axes[0, 0]
    for model in price_data.keys():
        model_prices = price_data[model]
        ax1.hist(model_prices, alpha=0.6, label=model, bins=20, density=True)
        # Add KDE curve
        try:
            kde = stats.gaussian_kde(model_prices)
            x_range = np.linspace(min(model_prices), max(model_prices), 100)
            ax1.plot(x_range, kde(x_range), linewidth=2)
        except:
            pass
    
    ax1.set_title('Price Distribution (Histogram + KDE)')
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box Plot
    ax2 = axes[0, 1]
    df.boxplot(column='price', by='model', ax=ax2)
    ax2.set_title('Price Distribution (Box Plot)')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Price')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Violin Plot
    ax3 = axes[0, 2]
    sns.violinplot(data=df, x='model', y='price', ax=ax3)
    ax3.set_title('Price Distribution (Violin Plot)')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Price')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Statistical Summary Table
    ax4 = axes[1, 0]
    ax4.axis('off')
    
    # Calculate statistics
    stats_data = []
    for model, prices in price_data.items():
        prices_array = np.array(prices)
        stats_row = [
            model,
            f"{np.mean(prices_array):.2f}",
            f"{np.median(prices_array):.2f}",
            f"{np.std(prices_array):.2f}",
            f"{np.min(prices_array):.2f}",
            f"{np.max(prices_array):.2f}",
            f"{len(prices_array)}"
        ]
        stats_data.append(stats_row)
    
    # Create table
    table = ax4.table(cellText=stats_data,
                      colLabels=['Model', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Count'],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax4.set_title('Statistical Summary', pad=20)
    
    # 5. Q-Q Plot (Normal distribution comparison)
    ax5 = axes[1, 1]
    colors = plt.cm.Set1(np.linspace(0, 1, len(price_data)))
    
    for i, (model, prices) in enumerate(price_data.items()):
        stats.probplot(prices, dist="norm", plot=ax5)
        # Customize the last plotted line
        ax5.get_lines()[-1].set_color(colors[i])
        ax5.get_lines()[-1].set_label(model)
        ax5.get_lines()[-2].set_color(colors[i])
    
    ax5.set_title('Q-Q Plot (Normal Distribution)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative Distribution Function
    ax6 = axes[1, 2]
    for model, prices in price_data.items():
        sorted_prices = np.sort(prices)
        y = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)
        ax6.plot(sorted_prices, y, marker='o', markersize=3, label=model, linewidth=2)
    
    ax6.set_title('Cumulative Distribution Function')
    ax6.set_xlabel('Price')
    ax6.set_ylabel('Cumulative Probability')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # log statistics
    logger.debug("="*60)
    logger.debug("PRICE DISTRIBUTION ANALYSIS SUMMARY")
    logger.debug("="*60)
    
    for model, prices in price_data.items():
        prices_array = np.array(prices)
        logger.debug(f"\n{model.upper()}:")
        logger.debug(f"  • Count: {len(prices_array)}")
        logger.debug(f"  • Mean: ${np.mean(prices_array):.2f}")
        logger.debug(f"  • Median: ${np.median(prices_array):.2f}")
        logger.debug(f"  • Standard Deviation: ${np.std(prices_array):.2f}")
        logger.debug(f"  • Range: ${np.min(prices_array):.2f} - ${np.max(prices_array):.2f}")
        logger.debug(f"  • IQR: ${np.percentile(prices_array, 25):.2f} - ${np.percentile(prices_array, 75):.2f}")
        
        if len(prices_array) >= 3:
            _, p_value = stats.shapiro(prices_array)
            normal_status = "Normal" if p_value > 0.05 else "Not Normal"
            logger.debug(f"  • Distribution: {normal_status} (Shapiro-Wilk p={p_value:.4f})")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.debug(f"\nPlot saved to: {save_path}")
    
    return fig

