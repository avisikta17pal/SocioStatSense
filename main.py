#!/usr/bin/env python3
"""
Main entry point for the Adaptive Real-Time Statistical Modeling Framework.

This script provides multiple ways to run the application:
1. Interactive dashboard (default)
2. Data ingestion pipeline
3. Model training and evaluation
4. Batch analysis and reporting
"""

import asyncio
import click
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

# Import framework components
from src.data.ingestion import DataIngestionPipeline
from src.data.preprocessing import DataPreprocessor
from src.models.baseline_model import BaselineRegressionModel
from src.models.adaptive_model import AdaptiveBayesianModel
from src.models.change_point_detector import ChangePointDetector
from src.causal.granger_causality import GrangerCausalityAnalyzer
from src.causal.intervention_analysis import InterventionAnalyzer
from src.utils.config import load_config
from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", log_file="logs/app.log")
logger = get_logger(__name__)


@click.group()
@click.option('--config', '-c', default='configs/data_sources.yaml', 
              help='Path to configuration file')
@click.option('--log-level', default='INFO', 
              help='Logging level (DEBUG, INFO, WARNING, ERROR)')
@click.pass_context
def cli(ctx, config, log_level):
    """Adaptive Real-Time Statistical Modeling Framework for Socio-Economic Impact Analysis."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['log_level'] = log_level
    
    # Setup logging with specified level
    setup_logging(log_level=log_level, log_file="logs/app.log")
    
    logger.info("🚀 Starting Adaptive Socio-Economic Modeling Framework")


@cli.command()
@click.option('--port', '-p', default=8501, help='Port to run the dashboard on')
@click.option('--host', '-h', default='localhost', help='Host to run the dashboard on')
@click.pass_context
def dashboard(ctx, port, host):
    """Launch the interactive Streamlit dashboard."""
    logger.info(f"Launching dashboard on {host}:{port}")
    
    try:
        # Run Streamlit dashboard
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "src/dashboard/main_dashboard.py",
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")


@cli.command()
@click.option('--hours-back', default=24, help='Hours of historical data to fetch')
@click.option('--continuous', is_flag=True, help='Run continuous data ingestion')
@click.option('--interval', default=60, help='Update interval in minutes for continuous mode')
@click.pass_context
def ingest(ctx, hours_back, continuous, interval):
    """Run data ingestion pipeline."""
    logger.info("Starting data ingestion pipeline")
    
    try:
        # Load configuration
        config = load_config(Path(ctx.obj['config_path']))
        
        # Initialize pipeline
        pipeline = DataIngestionPipeline(config)
        
        if continuous:
            logger.info(f"Starting continuous ingestion with {interval} minute intervals")
            asyncio.run(pipeline.run_continuous_ingestion(interval_minutes=interval))
        else:
            # Single ingestion run
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)
            
            logger.info(f"Fetching data from {start_date} to {end_date}")
            
            # Fetch and merge data
            data_dict = asyncio.run(pipeline.fetch_all_data(start_date, end_date))
            merged_df = pipeline.merge_data_sources(data_dict)
            
            if not merged_df.empty:
                pipeline.save_to_database(merged_df)
                logger.info(f"Successfully ingested {len(merged_df)} records")
                print(f"✅ Data ingestion completed: {len(merged_df)} records saved")
            else:
                logger.warning("No data was fetched")
                print("⚠️  No data was fetched - check API configurations")
        
    except KeyboardInterrupt:
        logger.info("Data ingestion stopped by user")
    except Exception as e:
        logger.error(f"Error in data ingestion: {str(e)}")
        print(f"❌ Error: {str(e)}")


@cli.command()
@click.option('--model-type', default='baseline', 
              type=click.Choice(['baseline', 'bayesian', 'both']),
              help='Type of model to train')
@click.option('--target-vars', default='unemployment_rate,cpi,fed_funds_rate',
              help='Comma-separated list of target variables')
@click.option('--save-model', is_flag=True, help='Save trained model to disk')
@click.pass_context
def train(ctx, model_type, target_vars, save_model):
    """Train statistical models."""
    logger.info(f"Training {model_type} model(s)")
    
    try:
        # Load configuration and data
        config = load_config(Path(ctx.obj['config_path']))
        pipeline = DataIngestionPipeline(config)
        
        # Load recent data
        df = pipeline.get_latest_data(hours_back=168)  # Last week
        
        if df.empty:
            print("❌ No data available for training")
            return
        
        # Prepare data
        preprocessor = DataPreprocessor(config.preprocessing)
        target_variables = [var.strip() for var in target_vars.split(',')]
        
        # Create target variables and split data
        df_with_targets = preprocessor.create_target_variables(df, target_variables)
        X, y = preprocessor.split_features_targets(df_with_targets)
        
        # Remove rows with NaN targets
        valid_indices = ~y.isnull().all(axis=1)
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        if len(X_clean) == 0:
            print("❌ No valid training data available")
            return
        
        print(f"📊 Training data: {len(X_clean)} samples, {len(X_clean.columns)} features")
        
        # Train models
        if model_type in ['baseline', 'both']:
            print("🤖 Training baseline regression model...")
            baseline_model = BaselineRegressionModel()
            baseline_model.fit(X_clean, y_clean)
            
            # Evaluate model
            eval_results = baseline_model.evaluate(X_clean, y_clean)
            for target, metrics in eval_results.items():
                print(f"  {target}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
            
            if save_model:
                baseline_model.save_model("models/baseline_model.pkl")
                print("💾 Baseline model saved")
        
        if model_type in ['bayesian', 'both']:
            print("🧠 Training Bayesian hierarchical model...")
            bayesian_model = AdaptiveBayesianModel()
            bayesian_model.fit(X_clean, y_clean)
            
            # Get model summary
            model_summary = bayesian_model.get_model_summary()
            print(f"  Trained {model_summary['n_targets']} targets")
            
            if save_model:
                bayesian_model.save_model("models/bayesian_model.pkl")
                print("💾 Bayesian model saved")
        
        print("✅ Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        print(f"❌ Error: {str(e)}")


@cli.command()
@click.option('--output-dir', default='reports', help='Output directory for reports')
@click.option('--format', default='html', type=click.Choice(['html', 'pdf', 'json']),
              help='Report format')
@click.pass_context
def analyze(ctx, output_dir, format):
    """Run comprehensive analysis and generate reports."""
    logger.info("Starting comprehensive analysis")
    
    try:
        # Load configuration and data
        config = load_config(Path(ctx.obj['config_path']))
        pipeline = DataIngestionPipeline(config)
        
        # Load data
        df = pipeline.get_latest_data(hours_back=168)
        
        if df.empty:
            print("❌ No data available for analysis")
            return
        
        print(f"📊 Analyzing {len(df)} data points across {len(df.columns)} variables")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        change_detector = ChangePointDetector()
        granger_analyzer = GrangerCausalityAnalyzer()
        
        # 1. Change point analysis
        print("🔍 Detecting change points...")
        change_points = change_detector.detect_change_points(df)
        
        # 2. Causal analysis
        print("🔗 Analyzing causal relationships...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        analysis_vars = [col for col in numeric_cols if any(indicator in col.lower() 
                        for indicator in ['unemployment', 'cpi', 'gdp', 'fed_funds'])][:6]
        
        causality_results = granger_analyzer.analyze_pairwise_causality(df, analysis_vars)
        
        # 3. Generate comprehensive report
        print("📝 Generating analysis report...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'n_observations': len(df),
                'n_variables': len(df.columns),
                'time_range': {
                    'start': df.index[0].isoformat(),
                    'end': df.index[-1].isoformat()
                },
                'data_quality': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            'change_points': change_points,
            'causal_analysis': causality_results,
            'causal_summary': granger_analyzer.get_causal_summary_report() if causality_results else {}
        }
        
        # Save report
        if format == 'json':
            import json
            report_file = output_path / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Analysis completed! Report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        print(f"❌ Error: {str(e)}")


@cli.command()
@click.pass_context
def demo(ctx):
    """Run a quick demo with sample data."""
    logger.info("Running demo mode")
    
    try:
        print("🎬 Starting demo of Adaptive Socio-Economic Modeling Framework")
        print("=" * 60)
        
        # Load configuration
        config = load_config()
        
        # 1. Data ingestion demo
        print("\n📊 1. Data Ingestion Demo")
        pipeline = DataIngestionPipeline(config)
        
        # Fetch sample data
        data_dict = asyncio.run(pipeline.fetch_all_data())
        df = pipeline.merge_data_sources(data_dict)
        
        if not df.empty:
            print(f"✅ Fetched data: {len(df)} observations, {len(df.columns)} variables")
            pipeline.save_to_database(df)
        else:
            print("⚠️  Using mock data for demo")
        
        # 2. Preprocessing demo
        print("\n🔧 2. Data Preprocessing Demo")
        preprocessor = DataPreprocessor(config.preprocessing)
        
        if not df.empty:
            df_processed = preprocessor.clean_and_preprocess(df)
            print(f"✅ Preprocessing completed: {len(df_processed.columns)} features created")
        
        # 3. Model training demo
        print("\n🤖 3. Model Training Demo")
        if not df.empty:
            # Create targets
            target_vars = ['unemployment_rate', 'cpi'] if 'unemployment_rate' in df.columns else list(df.columns)[:2]
            df_with_targets = preprocessor.create_target_variables(df, target_vars)
            X, y = preprocessor.split_features_targets(df_with_targets)
            
            # Clean data
            valid_indices = ~y.isnull().all(axis=1)
            X_clean = X[valid_indices]
            y_clean = y[valid_indices]
            
            if len(X_clean) > 10:
                # Train baseline model
                baseline_model = BaselineRegressionModel()
                baseline_model.fit(X_clean, y_clean)
                print(f"✅ Baseline model trained on {len(X_clean)} samples")
                
                # Generate forecast
                forecast = baseline_model.forecast(X_clean.tail(1), steps=24)
                print(f"✅ Generated 24-hour forecast for {len(forecast)} targets")
        
        # 4. Causal analysis demo
        print("\n🔗 4. Causal Analysis Demo")
        if not df.empty:
            granger_analyzer = GrangerCausalityAnalyzer()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
            
            if len(numeric_cols) >= 2:
                causality_results = granger_analyzer.analyze_pairwise_causality(df, numeric_cols)
                if causality_results:
                    summary = granger_analyzer.get_causal_summary_report()
                    n_relationships = summary.get('summary_statistics', {}).get('significant_relationships', 0)
                    print(f"✅ Causal analysis completed: {n_relationships} significant relationships found")
        
        # 5. Change point detection demo
        print("\n📊 5. Change Point Detection Demo")
        if not df.empty:
            change_detector = ChangePointDetector()
            change_points = change_detector.detect_change_points(df)
            
            total_cps = sum(len(cps) for cps in change_points.values())
            print(f"✅ Change point detection completed: {total_cps} change points detected")
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\n📱 To launch the interactive dashboard, run:")
        print("   python main.py dashboard")
        print("\n📚 For more options, run:")
        print("   python main.py --help")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        print(f"❌ Demo error: {str(e)}")


@cli.command()
@click.option('--interval', default=60, help='Update interval in minutes')
@click.pass_context
def monitor(ctx, interval):
    """Run continuous monitoring and alerting."""
    logger.info(f"Starting monitoring with {interval} minute intervals")
    
    try:
        config = load_config(Path(ctx.obj['config_path']))
        pipeline = DataIngestionPipeline(config)
        change_detector = ChangePointDetector()
        
        print(f"🔍 Starting continuous monitoring (updates every {interval} minutes)")
        print("Press Ctrl+C to stop")
        
        while True:
            try:
                # Fetch latest data
                df = pipeline.get_latest_data(hours_back=24)
                
                if not df.empty:
                    # Check for change points
                    change_points = change_detector.detect_change_points(df)
                    alerts = change_detector.get_change_point_alerts(confidence_threshold=0.8)
                    
                    if alerts:
                        print(f"\n🚨 {len(alerts)} high-confidence alerts detected:")
                        for alert in alerts[:3]:  # Show top 3
                            print(f"  - {alert['variable']}: Change at {alert['change_point_timestamp']} "
                                  f"(confidence: {alert['confidence']:.2f})")
                    else:
                        print(f"✅ {datetime.now().strftime('%H:%M:%S')} - No alerts detected")
                
                # Wait for next interval
                asyncio.run(asyncio.sleep(interval * 60))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                asyncio.run(asyncio.sleep(60))  # Wait 1 minute before retrying
        
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring: {str(e)}")
        print(f"❌ Error: {str(e)}")


@cli.command()
@click.pass_context
def status(ctx):
    """Check system status and health."""
    print("🔍 Checking system status...")
    
    try:
        # Check configuration
        config = load_config(Path(ctx.obj['config_path']))
        print("✅ Configuration loaded successfully")
        
        # Check data availability
        pipeline = DataIngestionPipeline(config)
        data_summary = pipeline.get_data_summary()
        
        if data_summary:
            print(f"✅ Database contains data from {len(data_summary)} sources")
            for source, info in data_summary.items():
                latest = pd.to_datetime(info['latest_data'])
                hours_ago = (datetime.now() - latest).total_seconds() / 3600
                print(f"   - {source}: {info['record_count']} records (last update: {hours_ago:.1f}h ago)")
        else:
            print("⚠️  No data found in database")
        
        # Check model files
        model_files = list(Path("models").glob("*.pkl")) if Path("models").exists() else []
        if model_files:
            print(f"✅ Found {len(model_files)} saved model(s)")
            for model_file in model_files:
                print(f"   - {model_file.name}")
        else:
            print("⚠️  No saved models found")
        
        # Check logs
        log_files = list(Path("logs").glob("*.log")) if Path("logs").exists() else []
        if log_files:
            print(f"✅ Log files available: {len(log_files)}")
        
        print("\n🎯 System Status: Ready")
        
    except Exception as e:
        print(f"❌ System Status: Error - {str(e)}")


def main():
    """Main entry point - run dashboard by default."""
    if len(sys.argv) == 1:
        # No arguments provided - run dashboard
        print("🚀 Launching Adaptive Socio-Economic Modeling Dashboard...")
        print("📱 Opening dashboard at http://localhost:8501")
        print("Press Ctrl+C to stop")
        
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "src/dashboard/main_dashboard.py",
                "--server.port", "8501",
                "--server.address", "localhost"
            ]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped")
    else:
        # Run CLI
        cli()


if __name__ == "__main__":
    main()
