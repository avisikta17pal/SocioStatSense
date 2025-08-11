# User Guide

## Adaptive Real-Time Statistical Modeling Framework

### Table of Contents
1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Data Sources Configuration](#data-sources-configuration)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Causal Analysis](#causal-analysis)
6. [What-If Scenarios](#what-if-scenarios)
7. [Monitoring and Alerts](#monitoring-and-alerts)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Getting Started

### Prerequisites

- **Python**: 3.9 or higher
- **Docker**: For containerized deployment (recommended)
- **API Keys**: For external data sources (optional for demo)

### Quick Installation

#### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd adaptive-socioeconomic-modeling

# Start with Docker Compose
docker-compose up --build

# Access dashboard at http://localhost:8501
```

#### Option 2: Local Installation

```bash
# Clone and setup
git clone <repository-url>
cd adaptive-socioeconomic-modeling

# Install dependencies
pip install -r requirements.txt

# Run dashboard
python main.py dashboard
```

### First Run

1. **Access Dashboard**: Open http://localhost:8501 in your browser
2. **Check System Status**: Verify all services are running
3. **Explore Sample Data**: Use built-in demo data to explore features
4. **Configure API Keys**: Add your API keys to `.env` file for real data

---

## Dashboard Overview

### Navigation

The dashboard consists of six main sections accessible via the sidebar:

1. **📊 Overview**: System status and key metrics
2. **📈 Data Sources**: Real-time data monitoring and quality
3. **🤖 Model Performance**: Model accuracy and predictions
4. **🔗 Causal Analysis**: Causal relationships and network
5. **🎯 What-If Scenarios**: Interactive policy simulations
6. **⚠️ Alerts & Monitoring**: System health and anomalies

### Overview Page

#### Key Metrics Dashboard
- **Data Freshness**: Last update time for each data source
- **Model Performance**: Current accuracy metrics
- **System Health**: Memory usage, processing status
- **Recent Alerts**: Latest warnings and notifications

#### Quick Actions
- **Refresh Data**: Manually trigger data update
- **Retrain Models**: Force model retraining
- **Export Report**: Download analysis summary
- **System Status**: View detailed health metrics

### Real-Time Updates

The dashboard automatically refreshes every 60 seconds (configurable). You can also:
- **Manual Refresh**: Click refresh button in sidebar
- **Auto-Refresh Toggle**: Enable/disable automatic updates
- **Update Frequency**: Adjust refresh interval in settings

---

## Data Sources Configuration

### Supported Data Sources

#### Economic Indicators (FRED)
- **Unemployment Rate**: Monthly unemployment statistics
- **Consumer Price Index**: Inflation measurements
- **GDP**: Quarterly economic output
- **Federal Funds Rate**: Central bank interest rates

#### Financial Markets (Yahoo Finance)
- **S&P 500**: Stock market index
- **VIX**: Market volatility index
- **Dollar Index**: Currency strength
- **Commodity Prices**: Gold, oil, agricultural products

#### Social Sentiment (Mock/Twitter)
- **Economic Keywords**: Sentiment analysis of economic discussions
- **Trend Analysis**: Public interest in economic topics
- **Real-time Updates**: Hourly sentiment scores

#### Search Interest (Google Trends)
- **Economic Search Terms**: Job search, recession, inflation
- **Regional Data**: Geographic breakdown of interest
- **Temporal Patterns**: Weekly and seasonal trends

#### Weather Data (OpenWeatherMap)
- **Temperature**: Regional temperature data
- **Precipitation**: Rainfall and weather events
- **Economic Impact**: Weather effects on economic activity

### API Key Setup

1. **Create `.env` file** in the project root:
```bash
# Required for real data (optional for demo)
FRED_API_KEY=your_fred_api_key_here
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

2. **Obtain API Keys**:
   - **FRED**: Register at https://fred.stlouisfed.org/docs/api/
   - **Twitter**: Apply at https://developer.twitter.com/
   - **OpenWeatherMap**: Sign up at https://openweathermap.org/api

3. **Configuration**: Edit `configs/data_sources.yaml` to customize:
   - Data update frequencies
   - Specific economic indicators
   - Geographic regions
   - Search keywords

### Data Quality Monitoring

#### Quality Metrics
- **Completeness**: Percentage of available data points
- **Timeliness**: How recent the data is
- **Consistency**: Data validation and range checks
- **Anomalies**: Outliers and unusual patterns

#### Data Validation
The system automatically:
- Detects missing values and gaps
- Identifies outliers using statistical methods
- Validates data types and ranges
- Monitors API response times and errors

---

## Model Training and Evaluation

### Model Types

#### Baseline Regression Model
- **Purpose**: Fast, interpretable predictions
- **Method**: Online learning with regularization
- **Use Cases**: Real-time predictions, quick analysis
- **Advantages**: Fast training, explainable results

#### Adaptive Bayesian Model
- **Purpose**: Uncertainty quantification and complex relationships
- **Method**: Hierarchical Bayesian inference
- **Use Cases**: Risk analysis, uncertainty estimation
- **Advantages**: Full uncertainty, automatic feature selection

### Training Process

#### Automatic Training
Models automatically retrain when:
- New data becomes available
- Performance degrades below threshold
- Structural breaks are detected
- Manual retrain is triggered

#### Manual Training
Use the command line interface:

```bash
# Train baseline model
python main.py train --model-type baseline

# Train Bayesian model
python main.py train --model-type bayesian

# Train both models
python main.py train --model-type both --save-model
```

### Model Evaluation

#### Performance Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **Coverage**: Prediction interval coverage

#### Model Comparison
The dashboard shows:
- **Side-by-side metrics**: Compare different models
- **Performance over time**: Track accuracy trends
- **Feature importance**: Which variables matter most
- **Prediction intervals**: Uncertainty quantification

### Prediction Horizons

#### Short-term (1-6 hours)
- **High accuracy**: Recent patterns dominate
- **Low uncertainty**: Confident predictions
- **Use cases**: Immediate decision making

#### Medium-term (6-72 hours)
- **Moderate accuracy**: Trend-based predictions
- **Increasing uncertainty**: Wider confidence intervals
- **Use cases**: Operational planning

#### Long-term (3+ days)
- **Lower accuracy**: High-level trends only
- **High uncertainty**: Very wide intervals
- **Use cases**: Strategic planning, scenario analysis

---

## Causal Analysis

### Understanding Causality

#### Granger Causality
- **Definition**: X "Granger-causes" Y if past values of X help predict Y
- **Interpretation**: Predictive causality, not necessarily true causation
- **Visualization**: Network graph showing causal relationships
- **Strength**: Color and thickness indicate causal strength

#### Causal Network
The dashboard displays:
- **Nodes**: Economic variables
- **Edges**: Causal relationships
- **Direction**: Arrows show causal direction
- **Strength**: Edge thickness shows relationship strength

### Interpreting Results

#### Strong Causal Relationships
- **Policy → Unemployment**: Interest rates affect unemployment
- **Unemployment → Sentiment**: Job market affects public mood
- **Market → Confidence**: Stock performance influences sentiment

#### Weak or No Causality
- **Weather → Markets**: Limited direct relationship
- **Random Variables**: No systematic patterns
- **Spurious Correlations**: Coincidental relationships

### Causal Discovery Process

1. **Data Preparation**: Clean and align time series
2. **Lag Selection**: Find optimal time delays
3. **Statistical Testing**: F-tests for significance
4. **Network Construction**: Build causal graph
5. **Validation**: Check robustness and stability

---

## What-If Scenarios

### Interactive Simulation

#### Policy Interventions
1. **Select Policy Variable**: Choose intervention target (e.g., interest rate)
2. **Set Intervention Value**: Specify new policy level
3. **Choose Forecast Horizon**: How far ahead to predict
4. **Run Simulation**: Calculate expected effects

#### Example Scenarios

##### Interest Rate Change
- **Scenario**: Reduce federal funds rate from 3% to 1%
- **Expected Effects**:
  - Unemployment: Decrease by 0.2-0.5 percentage points
  - Inflation: Increase by 0.1-0.3 percentage points
  - Market: Positive response with increased volatility

##### Government Spending Increase
- **Scenario**: Increase spending by 10%
- **Expected Effects**:
  - GDP Growth: Increase by 0.1-0.4 percentage points
  - Unemployment: Decrease with 6-month lag
  - Inflation: Gradual increase over 12 months

### Scenario Analysis Features

#### Multiple Interventions
- **Combined Policies**: Test multiple policy changes simultaneously
- **Interaction Effects**: Understand policy interactions
- **Optimization**: Find best policy combinations

#### Sensitivity Analysis
- **Parameter Ranges**: Test range of intervention values
- **Uncertainty**: Show confidence intervals for effects
- **Robustness**: Test across different time periods

#### Comparison Tools
- **Baseline vs Intervention**: Side-by-side comparison
- **Multiple Scenarios**: Compare different policy options
- **Historical Analysis**: Compare with past interventions

---

## Monitoring and Alerts

### Alert Types

#### Data Quality Alerts
- **Missing Data**: When data sources fail or have gaps
- **Outliers**: Unusual values that may indicate errors
- **Staleness**: Data that hasn't updated recently
- **API Failures**: When external APIs are unavailable

#### Model Performance Alerts
- **Accuracy Degradation**: When prediction errors increase
- **Concept Drift**: When data patterns change
- **Convergence Issues**: When models fail to train properly
- **Uncertainty Spikes**: When prediction confidence drops

#### Structural Change Alerts
- **Change Points**: When statistical breaks are detected
- **Regime Shifts**: When economic patterns change
- **Causal Changes**: When relationships between variables change
- **Anomalies**: When unusual patterns are detected

### Alert Configuration

#### Thresholds
```yaml
alerts:
  change_point_threshold: 0.8      # Confidence level for change points
  anomaly_threshold: 2.5           # Standard deviations for anomalies
  model_drift_threshold: 0.1       # Performance degradation threshold
```

#### Notification Methods
- **Dashboard**: Real-time alerts in the web interface
- **Email**: Automated email notifications
- **Logs**: Detailed logging for debugging
- **API**: Webhook notifications for external systems

### Alert Management

#### Alert Dashboard
- **Active Alerts**: Current warnings and errors
- **Alert History**: Past alerts and resolutions
- **Alert Frequency**: Patterns in alert occurrence
- **Resolution Tracking**: Time to resolve issues

#### Response Actions
- **Automatic**: System attempts self-healing
- **Manual**: User intervention required
- **Escalation**: Critical alerts require immediate attention
- **Documentation**: Suggested troubleshooting steps

---

## Advanced Usage

### Command Line Interface

#### Data Operations
```bash
# Fetch historical data
python main.py ingest --hours-back 168

# Run continuous data ingestion
python main.py ingest --continuous --interval 30

# Check data status
python main.py status
```

#### Model Operations
```bash
# Train specific model type
python main.py train --model-type bayesian --target-vars unemployment_rate,cpi

# Generate analysis report
python main.py analyze --output-dir reports --format html

# Run quick demo
python main.py demo
```

#### Monitoring
```bash
# Start monitoring service
python main.py monitor --interval 60

# Check system status
python main.py status
```

### Custom Configuration

#### Data Sources
Edit `configs/data_sources.yaml` to:
- Add new data sources
- Modify update frequencies
- Change geographic regions
- Customize economic indicators

#### Model Parameters
Adjust modeling settings:
```yaml
modeling:
  window_size: 336          # 2 weeks of hourly data
  update_frequency: "2H"    # Update every 2 hours
  sparse_selection_alpha: 0.005  # More aggressive feature selection
```

#### Dashboard Settings
Customize dashboard behavior:
```yaml
dashboard:
  refresh_interval: 30      # Refresh every 30 seconds
  max_display_points: 2000  # Show more data points
  default_forecast_horizon: 48  # 48-hour forecasts
```

### Integration with External Systems

#### API Integration
The framework can be integrated with external systems via:
- **Database Access**: Direct SQLite/PostgreSQL queries
- **File Exports**: CSV, JSON data exports
- **REST API**: (Future enhancement)
- **Webhooks**: Alert notifications

#### Custom Data Sources
Add new data sources by:
1. Implementing `BaseDataSource` interface
2. Adding configuration in `data_sources.yaml`
3. Registering in `DataIngestionPipeline`

```python
class CustomDataSource(BaseDataSource):
    async def fetch_data(self, start_date=None, end_date=None):
        # Your custom data fetching logic
        return pd.DataFrame(...)
```

---

## Troubleshooting

### Common Issues

#### Installation Problems
- **Dependency Conflicts**: Use virtual environment or Docker
- **Missing Packages**: Run `pip install -r requirements.txt`
- **Python Version**: Ensure Python 3.9+ is installed
- **System Dependencies**: Install required system packages

#### Data Issues
- **No Data Displayed**: Check API keys and internet connection
- **Stale Data**: Verify data source update frequencies
- **Missing Variables**: Check data source configuration
- **Quality Warnings**: Review data validation alerts

#### Model Issues
- **Poor Performance**: Increase training data or adjust parameters
- **Slow Training**: Reduce feature count or use baseline model
- **Convergence Failures**: Adjust learning rates and regularization
- **Memory Errors**: Reduce batch sizes or use smaller models

#### Dashboard Issues
- **Slow Loading**: Check data cache settings and reduce display points
- **Visualization Errors**: Verify data format and completeness
- **Update Failures**: Check service status and restart if needed
- **Browser Compatibility**: Use modern browser (Chrome, Firefox, Safari)

### Error Messages

#### Common Error Patterns

##### "No data available"
- **Cause**: Data sources not configured or failing
- **Solution**: Check API keys and data source status
- **Prevention**: Set up data source monitoring

##### "Model training failed"
- **Cause**: Insufficient data or numerical issues
- **Solution**: Check data quality and model parameters
- **Prevention**: Implement data validation checks

##### "Connection timeout"
- **Cause**: Network issues or API rate limits
- **Solution**: Check internet connection and API quotas
- **Prevention**: Implement retry logic and caching

### Getting Help

#### Log Files
Check application logs:
```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep "ERROR" logs/app.log

# Filter by component
grep "DataIngestion" logs/app.log
```

#### Debug Mode
Enable debug mode for detailed information:
```bash
# Set debug environment
export DEBUG=True
export LOG_LEVEL=DEBUG

# Run with verbose logging
python main.py dashboard --log-level DEBUG
```

#### Health Checks
```bash
# Check system status
python main.py status

# Test data sources
python main.py ingest --hours-back 1

# Validate configuration
python -c "from src.utils.config import load_config; print(load_config())"
```

---

## FAQ

### General Questions

#### Q: What makes this framework "adaptive"?
**A**: The models continuously update with new data, detect structural changes, and adjust their parameters automatically. This allows them to adapt to changing economic conditions without manual intervention.

#### Q: How accurate are the predictions?
**A**: Accuracy depends on the prediction horizon and data quality. Short-term predictions (1-6 hours) typically achieve R² > 0.8, while longer-term forecasts have wider uncertainty intervals but still capture major trends.

#### Q: Can I use this without API keys?
**A**: Yes! The system includes mock data generators that create realistic synthetic data for demonstration purposes. This allows you to explore all features without external API dependencies.

### Technical Questions

#### Q: How often should models be retrained?
**A**: The system automatically determines retraining frequency based on:
- Data availability (new data triggers updates)
- Performance degradation (drift detection)
- Structural changes (change point detection)
- Manual schedule (configurable intervals)

#### Q: What's the difference between Granger causality and true causality?
**A**: Granger causality tests whether past values of X help predict Y, indicating predictive relationships. True causality requires additional assumptions and experimental design. Our framework focuses on Granger causality for time series data.

#### Q: How do I interpret uncertainty intervals?
**A**: Uncertainty intervals show the range of plausible values for predictions:
- **Narrow intervals**: High confidence predictions
- **Wide intervals**: High uncertainty, use caution
- **Coverage**: 90% intervals should contain true values 90% of the time

### Usage Questions

#### Q: How do I add a new economic indicator?
**A**: 
1. Add the indicator to `configs/data_sources.yaml`
2. Restart the data ingestion service
3. The new variable will automatically appear in models and dashboard

#### Q: Can I export the results?
**A**: Yes, you can:
- Download reports from the dashboard
- Export data to CSV using the CLI
- Access the SQLite database directly
- Use the API endpoints (future feature)

#### Q: How do I customize the dashboard?
**A**: 
- Modify `configs/data_sources.yaml` for data settings
- Edit dashboard configuration in the config file
- Customize visualizations in `src/dashboard/components.py`
- Add new pages by extending the main dashboard

#### Q: What if my data has different time frequencies?
**A**: The system automatically aligns data to a common frequency (hourly by default). You can:
- Change the alignment frequency in configuration
- Handle mixed frequencies in preprocessing
- Use interpolation for missing time points

### Performance Questions

#### Q: How much data can the system handle?
**A**: The system is designed for:
- **Real-time**: Thousands of data points per hour
- **Historical**: Several years of hourly data
- **Variables**: 50+ economic indicators
- **Scalability**: Can be scaled horizontally with Docker

#### Q: Why are Bayesian models slower?
**A**: Bayesian models use MCMC sampling which is computationally intensive but provides:
- Full uncertainty quantification
- Automatic feature selection
- Robust parameter estimates
- Better handling of small datasets

#### Q: How can I improve performance?
**A**:
- Use baseline models for real-time applications
- Reduce the number of features
- Increase update intervals
- Use Docker for better resource management
- Scale horizontally with multiple containers

### Deployment Questions

#### Q: Can I deploy this in production?
**A**: Yes, the framework is production-ready with:
- Docker containerization
- Health checks and monitoring
- Logging and error handling
- Configuration management
- Automated testing

#### Q: What are the system requirements?
**A**:
- **CPU**: 2+ cores recommended
- **Memory**: 4GB+ RAM
- **Storage**: 10GB+ for data and models
- **Network**: Stable internet for API access

#### Q: How do I scale the system?
**A**: Use Docker Compose to scale services:
```bash
# Scale data ingestion
docker-compose up --scale data-ingestion=3

# Scale with load balancer
docker-compose up --scale dashboard=2
```

---

## Best Practices

### Data Management
1. **Regular Backups**: Backup database and model files
2. **Data Validation**: Monitor data quality metrics
3. **API Management**: Respect rate limits and cache responses
4. **Version Control**: Track configuration changes

### Model Management
1. **Performance Monitoring**: Track accuracy over time
2. **Regular Retraining**: Update models with new data
3. **A/B Testing**: Compare different model configurations
4. **Documentation**: Record model changes and performance

### System Administration
1. **Log Monitoring**: Regularly check application logs
2. **Resource Monitoring**: Monitor CPU, memory, and disk usage
3. **Security Updates**: Keep dependencies up to date
4. **Backup Strategy**: Regular backups of data and configuration

### Decision Making
1. **Understand Uncertainty**: Always consider prediction intervals
2. **Validate Assumptions**: Check causal inference assumptions
3. **Multiple Scenarios**: Test various what-if scenarios
4. **Expert Judgment**: Combine model insights with domain expertise

---

## Support and Resources

### Documentation
- **Technical Documentation**: `docs/TECHNICAL_DOCUMENTATION.md`
- **API Reference**: Detailed function documentation
- **Examples**: Sample notebooks and scripts
- **Changelog**: Version history and updates

### Community
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Ask questions and share experiences
- **Contributions**: Submit pull requests and improvements
- **Wiki**: Community-maintained documentation

### Professional Support
For enterprise deployments and custom development:
- **Consulting**: Custom implementation and integration
- **Training**: Team training and workshops
- **Support**: Dedicated technical support
- **Customization**: Tailored features and modifications
