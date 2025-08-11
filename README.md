# Adaptive Real-Time Statistical Modeling Framework for Socio-Economic Impact Analysis

An end-to-end, production-ready adaptive statistical modeling system that continuously ingests real-time multi-source socio-economic data streams, updates interpretable statistical models, detects structural shifts, performs causal inference, and provides actionable insights with uncertainty quantification.

## 🚀 Features

- **Real-Time Data Ingestion**: Multi-source data pipeline with robust preprocessing
- **Adaptive Bayesian Modeling**: Online learning with sparse variable selection
- **Change-Point Detection**: Automatic detection of structural breaks and regime shifts
- **Causal Inference**: Granger causality and DoWhy integration for causal relationships
- **Interactive Dashboard**: Real-time visualizations with explainability
- **Decision Support**: What-if simulations and automated report generation
- **Production Ready**: Dockerized, tested, and monitored

## 📊 Data Sources

- Government economic indicators (unemployment, inflation)
- Social media sentiment analysis
- Google Trends and web search interest
- Environmental and weather data
- Market and commodity prices

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, PyMC3/4, scikit-learn
- **Data Processing**: Pandas, NumPy, Requests
- **Modeling**: TensorFlow Probability, ruptures, causalml, DoWhy
- **Frontend**: Streamlit with Plotly visualizations
- **Deployment**: Docker, Docker Compose
- **Testing**: pytest, unittest

## 🏗️ Architecture

```
├── src/
│   ├── data/           # Data ingestion and preprocessing
│   ├── models/         # Statistical and ML models
│   ├── causal/         # Causal inference algorithms
│   ├── dashboard/      # Streamlit dashboard
│   └── utils/          # Utilities and helpers
├── tests/              # Comprehensive test suite
├── data/               # Sample and cached data
├── configs/            # Configuration files
├── docker/             # Docker configuration
└── docs/               # Documentation
```

## 🚀 Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd adaptive-socioeconomic-modeling
   pip install -r requirements.txt
   ```

2. **Run with Docker**:
   ```bash
   docker-compose up --build
   ```

3. **Access Dashboard**:
   Open http://localhost:8501 in your browser

## 📈 Usage

1. **Configure Data Sources**: Edit `configs/data_sources.yaml`
2. **Start Data Ingestion**: Run the data pipeline
3. **Monitor Models**: View real-time updates in the dashboard
4. **Run Simulations**: Use the what-if analysis interface
5. **Generate Reports**: Download automated insights and recommendations

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📚 Documentation

See `docs/` directory for detailed technical documentation and user guides.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
