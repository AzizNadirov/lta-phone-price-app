# 📱 Phone Price Predictor - educational pet project for you

**Is that phone really worth its price tag?** This app helps you find out!

Phone Price Predictor is a user-friendly tool that crawls phone specifications from any product URL, uses machine learning to predict what the phone *should* cost, and compares it with the actual listing price. Perfect for finding great deals or avoiding overpriced devices!

![Python Version](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen)

## ✨ Features

- **Simple URL Input**: Just paste any phone product page URL
- **Automatic Specification Extraction**: Pulls detailed specs through web crawling
- **Price Prediction**: Uses machine learning to determine fair market value
- **Price Comparison**: See how the listed price compares to predicted value
- **Visual Analytics**: Explore price distributions with beautiful charts
- **History Tracking**: Keep track of phones you've analyzed

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- Git (for cloning the repository)

### Installation

#### Option 1: Regular Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/phone-price-predictor.git
   cd phone-price-predictor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   
   On Windows:
   ```bash
   .venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

4. **Install UV package manager (optional)**
   ```bash
   pip install uv
   ```

5. **Install dependencies**
   
   Using UV:
   ```bash
   uv sync
   ```

6. **Install Playwright browsers**
   ```bash
   playwright install
   ```

#### Option 2: Docker Installation (Easiest)

1. **Clone the repository**
   ```bash
   git clone https://github.com/AzizNadirov/lta-phone-price-app.git
   cd phone-price-predictor
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up
   ```

That's it! The app will be available at http://localhost:8501

## 📱 How to Use

1. **Launch the application**
   ```bash
   streamlit run src/phoner.py
   ```
   
   Or if using Docker, it's already running at http://localhost:8501

2. **Enter a phone product URL**
   - Example: `https://www.gsmarena.com/samsung_galaxy_s23-12082.php`
   - The app works with popular sites like GSMArena, Amazon, Best Buy, etc.

3. **Click "Analyze Phone"**
   - The app will crawl the specifications
   - Predict the price based on the phone's features
   - Compare with the actual listing price
   - Show statistical visualizations

4. **Interpret the results**
   - If the predicted price is higher than the actual price, you might have found a good deal!
   - If the predicted price is lower, the phone might be overpriced for its specifications
   - Explore the distribution charts to see how this phone compares to similar models

5. **View your history**
   - Check the sidebar to see all phones you've analyzed during your session

## 🔧 Understanding the Visualizations

The app includes several visualizations to help you understand the phone's price positioning:

- **Histogram with KDE**: Shows how frequently phones with similar specifications appear at different price points
- **Box Plot**: Shows the median price and spread of prices for similar models
- **Violin Plot**: Shows the full distribution of prices, with wider sections indicating more common price points
- **Statistical Summary**: Shows key numbers like average price, median, and price range
- **Q-Q Plot**: Technical chart showing how well the prices follow a normal distribution
- **Cumulative Distribution**: Shows percentiles of prices (e.g., what percentage of similar phones cost less than a certain amount)

## 🗂️ Project Structure Simplified

```
.
├── src/                  # Main source code
│   ├── phoner.py         # Streamlit application (the main file)
│   ├── handlers.py       # Core functionality managers
│   ├── schemas.py        # Data validation models
│   ├── parsing/          # Web crawling code
│   ├── ml/               # Machine learning models
│   │   └── models/       # Trained prediction models
│   └── data/             # Phone dataset
├── logs/                 # Application logs
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
└── docker-compose.yaml   # Docker Compose configuration
```

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the app:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
