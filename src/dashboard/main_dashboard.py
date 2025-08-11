"""Main Streamlit dashboard for the adaptive statistical modeling framework."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import time

# Import framework components
from ..data.ingestion import DataIngestionPipeline
from ..models.baseline_model import BaselineRegressionModel
from ..models.adaptive_model import AdaptiveBayesianModel
from ..models.change_point_detector import ChangePointDetector
from ..causal.granger_causality import GrangerCausalityAnalyzer
from ..causal.intervention_analysis import InterventionAnalyzer
from ..utils.config import load_config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Adaptive Socio-Economic Modeling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .alert-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_cached():
    """Load data with caching."""
    try:
        config = load_config()
        pipeline = DataIngestionPipeline(config)
        
        # Try to load from database first
        df = pipeline.get_latest_data(hours_back=168)  # Last week
        
        if df.empty:
            # If no data in database, fetch new data
            data_dict = asyncio.run(pipeline.fetch_all_data())
            df = pipeline.merge_data_sources(data_dict)
            if not df.empty:
                pipeline.save_to_database(df)
        
        return df, pipeline
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), None


@st.cache_resource
def initialize_models():
    """Initialize models with caching."""
    try:
        config = load_config()
        
        # Initialize models
        baseline_model = BaselineRegressionModel()
        bayesian_model = AdaptiveBayesianModel()
        change_detector = ChangePointDetector()
        granger_analyzer = GrangerCausalityAnalyzer()
        intervention_analyzer = InterventionAnalyzer()
        
        return {
            'baseline': baseline_model,
            'bayesian': bayesian_model,
            'change_detector': change_detector,
            'granger': granger_analyzer,
            'intervention': intervention_analyzer,
            'config': config
        }
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        return {}


def render_sidebar():
    """Render the sidebar with navigation and controls."""
    st.sidebar.markdown("# 📊 Navigation")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Data Sources", "Model Performance", "Causal Analysis", "What-If Scenarios", "Alerts & Monitoring"]
    )
    
    st.sidebar.markdown("---")
    
    # Data refresh controls
    st.sidebar.markdown("## 🔄 Data Controls")
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)", value=True)
    
    if auto_refresh:
        time.sleep(60)
        st.rerun()
    
    # Model controls
    st.sidebar.markdown("## 🤖 Model Controls")
    
    model_type = st.sidebar.selectbox(
        "Active Model",
        ["Baseline Regression", "Bayesian Hierarchical"]
    )
    
    if st.sidebar.button("Retrain Models"):
        st.cache_resource.clear()
        st.success("Models will be retrained on next prediction")
    
    return page, model_type


def render_overview_page(df, models):
    """Render the overview page with key metrics and summaries."""
    st.markdown('<h1 class="main-header">🌍 Adaptive Socio-Economic Modeling Dashboard</h1>', 
                unsafe_allow_html=True)
    
    if df.empty:
        st.error("No data available. Please check data sources and API configurations.")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", f"{len(df):,}", delta=f"+{len(df.tail(24))}" if len(df) > 24 else None)
    
    with col2:
        st.metric("Variables Tracked", len(df.columns), delta=None)
    
    with col3:
        latest_update = df.index[-1] if not df.empty else "N/A"
        st.metric("Last Update", latest_update.strftime("%H:%M") if latest_update != "N/A" else "N/A")
    
    with col4:
        data_quality = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Quality", f"{data_quality:.1f}%", delta=None)
    
    # Main time series visualization
    st.markdown("## 📈 Key Economic Indicators")
    
    # Select key indicators to display
    key_indicators = [col for col in df.columns if any(indicator in col.lower() 
                     for indicator in ['unemployment', 'cpi', 'gdp', 'fed_funds'])]
    
    if key_indicators:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=key_indicators[:4],
            vertical_spacing=0.1
        )
        
        for i, indicator in enumerate(key_indicators[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Plot time series
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[indicator],
                    mode='lines',
                    name=indicator,
                    line=dict(width=2)
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent alerts and notifications
    st.markdown("## 🚨 Recent Alerts")
    
    # Mock alerts for demonstration
    alerts = [
        {"type": "warning", "message": "Change point detected in unemployment rate", "time": "2 hours ago"},
        {"type": "info", "message": "Model retrained with new data", "time": "4 hours ago"},
        {"type": "success", "message": "All data sources updated successfully", "time": "1 hour ago"}
    ]
    
    for alert in alerts:
        alert_class = f"alert-{alert['type']}" if alert['type'] != 'info' else "alert-success"
        st.markdown(
            f'<div class="alert-box {alert_class}"><strong>{alert["message"]}</strong> - {alert["time"]}</div>',
            unsafe_allow_html=True
        )


def render_data_sources_page(df, pipeline):
    """Render the data sources page."""
    st.markdown("# 📊 Data Sources")
    
    if pipeline is None:
        st.error("Data pipeline not available")
        return
    
    # Data source status
    st.markdown("## 📡 Source Status")
    
    data_summary = pipeline.get_data_summary()
    
    if data_summary:
        for source, info in data_summary.items():
            with st.expander(f"📈 {source.title()}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Records", f"{info['record_count']:,}")
                
                with col2:
                    st.metric("Data Span", f"{info['data_span_hours']:.1f} hours")
                
                with col3:
                    latest_data = pd.to_datetime(info['latest_data'])
                    hours_ago = (datetime.now() - latest_data).total_seconds() / 3600
                    st.metric("Last Update", f"{hours_ago:.1f}h ago")
    
    # Data quality metrics
    st.markdown("## 📊 Data Quality")
    
    if not df.empty:
        # Missing data heatmap
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        fig = px.bar(
            x=missing_pct.index,
            y=missing_pct.values,
            title="Missing Data Percentage by Variable",
            labels={'x': 'Variables', 'y': 'Missing %'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔗 Variable Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:15]  # Limit for visualization
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Raw data table
    st.markdown("## 📋 Recent Data")
    if not df.empty:
        st.dataframe(df.tail(20), use_container_width=True)


def render_model_performance_page(df, models, model_type):
    """Render the model performance page."""
    st.markdown("# 🤖 Model Performance")
    
    if df.empty:
        st.warning("No data available for model training")
        return
    
    # Model selection and training
    st.markdown("## 🎯 Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_variables = st.multiselect(
            "Select Target Variables",
            [col for col in df.columns if any(indicator in col.lower() 
             for indicator in ['unemployment', 'cpi', 'gdp', 'fed_funds'])],
            default=[col for col in df.columns if 'unemployment' in col.lower()][:1]
        )
    
    with col2:
        forecast_horizon = st.slider("Forecast Horizon (hours)", 1, 72, 24)
    
    if st.button("Train/Update Model") and target_variables:
        with st.spinner("Training model..."):
            try:
                # Prepare data for modeling
                from ..data.preprocessing import DataPreprocessor
                preprocessor = DataPreprocessor(models['config'].preprocessing)
                
                # Create target variables
                df_with_targets = preprocessor.create_target_variables(df, target_variables)
                X, y = preprocessor.split_features_targets(df_with_targets)
                
                # Remove rows with NaN targets
                valid_indices = ~y.isnull().all(axis=1)
                X_clean = X[valid_indices]
                y_clean = y[valid_indices]
                
                if len(X_clean) > 0:
                    # Train selected model
                    if model_type == "Baseline Regression":
                        model = models['baseline']
                        model.fit(X_clean, y_clean)
                        st.success(f"Baseline model trained with {len(X_clean)} samples")
                    else:
                        model = models['bayesian']
                        model.fit(X_clean, y_clean)
                        st.success(f"Bayesian model trained with {len(X_clean)} samples")
                    
                    # Store trained model in session state
                    st.session_state['trained_model'] = model
                    st.session_state['model_type'] = model_type
                    st.session_state['X_data'] = X_clean
                    st.session_state['y_data'] = y_clean
                else:
                    st.error("No valid training data available")
                    
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    # Model performance metrics
    if 'trained_model' in st.session_state:
        st.markdown("## 📊 Model Performance")
        
        model = st.session_state['trained_model']
        X_data = st.session_state['X_data']
        y_data = st.session_state['y_data']
        
        # Model summary
        model_summary = model.get_model_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Targets Modeled", model_summary['n_targets'])
        
        with col2:
            if 'latest_performance' in model_summary:
                r2 = model_summary['latest_performance'].get('r2', 0)
                st.metric("R² Score", f"{r2:.4f}")
        
        with col3:
            if 'latest_performance' in model_summary:
                rmse = model_summary['latest_performance'].get('rmse', 0)
                st.metric("RMSE", f"{rmse:.4f}")
        
        # Feature importance
        st.markdown("### 🎯 Feature Importance")
        
        feature_importance = model.get_feature_importance()
        
        for target_var, importances in feature_importance.items():
            st.markdown(f"#### {target_var}")
            
            # Create feature importance chart
            if isinstance(list(importances.values())[0], dict):
                # Bayesian model with uncertainty
                features = list(importances.keys())[:10]
                means = [importances[f]['mean_importance'] for f in features]
                stds = [importances[f]['std_importance'] for f in features]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=means,
                    y=features,
                    orientation='h',
                    error_x=dict(type='data', array=stds),
                    name='Importance'
                ))
            else:
                # Regular model
                features = list(importances.keys())[:10]
                values = [importances[f] for f in features]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    name='Importance'
                ))
            
            fig.update_layout(
                title=f"Top 10 Features for {target_var}",
                xaxis_title="Importance",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Predictions and forecasts
        st.markdown("### 🔮 Predictions & Forecasts")
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                try:
                    # Generate forecast
                    forecast_result = model.forecast(X_data.tail(1), steps=forecast_horizon)
                    
                    # Display forecasts
                    for target_var, forecast in forecast_result.items():
                        st.markdown(f"#### {target_var} Forecast")
                        
                        # Create forecast plot
                        future_times = pd.date_range(
                            start=df.index[-1] + pd.Timedelta(hours=1),
                            periods=forecast_horizon,
                            freq='H'
                        )
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=df.index[-72:],  # Last 3 days
                            y=df[target_var].tail(72) if target_var in df.columns else [0]*72,
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Forecast mean
                        fig.add_trace(go.Scatter(
                            x=future_times,
                            y=forecast['mean'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Uncertainty bands
                        if 'upper_95' in forecast and 'lower_95' in forecast:
                            fig.add_trace(go.Scatter(
                                x=future_times,
                                y=forecast['upper_95'],
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=future_times,
                                y=forecast['lower_95'],
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                name='95% CI',
                                fillcolor='rgba(255,0,0,0.2)'
                            ))
                        
                        fig.update_layout(
                            title=f"{target_var} Forecast ({forecast_horizon}h ahead)",
                            xaxis_title="Time",
                            yaxis_title=target_var,
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")


def render_causal_analysis_page(df, models):
    """Render the causal analysis page."""
    st.markdown("# 🔗 Causal Analysis")
    
    if df.empty:
        st.warning("No data available for causal analysis")
        return
    
    # Granger causality analysis
    st.markdown("## 🔍 Granger Causality Analysis")
    
    if st.button("Run Granger Causality Analysis"):
        with st.spinner("Analyzing causal relationships..."):
            try:
                granger_analyzer = models['granger']
                
                # Select variables for analysis
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                analysis_vars = [col for col in numeric_cols if any(indicator in col.lower() 
                               for indicator in ['unemployment', 'cpi', 'gdp', 'fed_funds', 'sentiment'])][:6]
                
                # Run analysis
                causality_results = granger_analyzer.analyze_pairwise_causality(df, analysis_vars)
                
                if causality_results:
                    # Display causality matrix
                    if granger_analyzer.causality_matrix is not None:
                        st.markdown("### 📊 Causality Matrix")
                        
                        fig = px.imshow(
                            granger_analyzer.causality_matrix,
                            text_auto=True,
                            aspect="auto",
                            title="Granger Causality Strength Matrix",
                            labels=dict(x="Effect Variable", y="Cause Variable", color="F-Statistic")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Network visualization
                    st.markdown("### 🕸️ Causal Network")
                    
                    network_metrics = granger_analyzer.get_causal_network_metrics()
                    if network_metrics:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Nodes", network_metrics['n_nodes'])
                        with col2:
                            st.metric("Causal Links", network_metrics['n_edges'])
                        with col3:
                            st.metric("Network Density", f"{network_metrics['density']:.3f}")
                    
                    # Strongest relationships
                    summary_report = granger_analyzer.get_causal_summary_report()
                    if 'strongest_relationships' in summary_report:
                        st.markdown("### 💪 Strongest Causal Relationships")
                        
                        relationships_df = pd.DataFrame(summary_report['strongest_relationships'])
                        if not relationships_df.empty:
                            st.dataframe(relationships_df, use_container_width=True)
                
                st.success("Granger causality analysis completed!")
                
            except Exception as e:
                st.error(f"Error in causal analysis: {str(e)}")


def render_intervention_page(df, models):
    """Render the what-if scenarios page."""
    st.markdown("# 🎛️ What-If Scenarios")
    
    if df.empty:
        st.warning("No data available for intervention analysis")
        return
    
    # Initialize intervention analyzer
    intervention_analyzer = models['intervention']
    
    if st.button("Initialize Causal Models"):
        with st.spinner("Fitting causal models..."):
            try:
                intervention_analyzer.fit_causal_models(df)
                st.session_state['intervention_ready'] = True
                st.success("Causal models fitted successfully!")
            except Exception as e:
                st.error(f"Error fitting causal models: {str(e)}")
    
    if st.session_state.get('intervention_ready', False):
        st.markdown("## 🎯 Policy Intervention Simulator")
        
        # Intervention controls
        col1, col2 = st.columns(2)
        
        with col1:
            intervention_var = st.selectbox(
                "Intervention Variable",
                [col for col in df.columns if any(indicator in col.lower() 
                 for indicator in ['fed_funds', 'unemployment', 'policy'])]
            )
        
        with col2:
            if intervention_var:
                current_value = df[intervention_var].iloc[-1] if intervention_var in df.columns else 0
                intervention_value = st.number_input(
                    f"New {intervention_var} Value",
                    value=float(current_value),
                    step=0.1
                )
        
        # Target variables to analyze
        target_variables = st.multiselect(
            "Variables to Analyze Impact On",
            [col for col in df.columns if col != intervention_var and any(indicator in col.lower() 
             for indicator in ['unemployment', 'cpi', 'gdp', 'sentiment'])],
            default=[col for col in df.columns if 'unemployment' in col.lower()][:2]
        )
        
        simulation_horizon = st.slider("Simulation Horizon (hours)", 1, 168, 72)
        
        if st.button("Run Intervention Simulation") and intervention_var and target_variables:
            with st.spinner("Running intervention simulation..."):
                try:
                    # Run intervention analysis
                    intervention_result = intervention_analyzer.simulate_intervention(
                        interventions={intervention_var: intervention_value},
                        forecast_horizon=simulation_horizon,
                        n_simulations=100
                    )
                    
                    # Display results
                    st.markdown("### 📈 Intervention Results")
                    
                    for target_var in target_variables:
                        if target_var in intervention_result['causal_effects']:
                            effects = intervention_result['causal_effects'][target_var]
                            
                            st.markdown(f"#### Impact on {target_var}")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                max_effect = effects['max_absolute_effect']
                                if isinstance(max_effect, np.ndarray):
                                    max_effect = np.max(max_effect)
                                st.metric("Max Effect", f"{max_effect:.4f}")
                            
                            with col2:
                                significance = "Yes" if effects['significant_effect'] else "No"
                                st.metric("Significant", significance)
                            
                            with col3:
                                peak_time = effects['time_to_peak_effect']
                                st.metric("Peak Effect Time", f"{peak_time}h")
                            
                            # Plot intervention effect over time
                            if target_var in intervention_result['intervention_forecast']:
                                forecast_data = intervention_result['intervention_forecast'][target_var]
                                baseline_data = intervention_result['baseline_forecast'][target_var]
                                
                                future_times = pd.date_range(
                                    start=df.index[-1] + pd.Timedelta(hours=1),
                                    periods=simulation_horizon,
                                    freq='H'
                                )
                                
                                fig = go.Figure()
                                
                                # Baseline scenario
                                fig.add_trace(go.Scatter(
                                    x=future_times,
                                    y=baseline_data['mean'],
                                    mode='lines',
                                    name='Baseline',
                                    line=dict(color='blue')
                                ))
                                
                                # Intervention scenario
                                fig.add_trace(go.Scatter(
                                    x=future_times,
                                    y=forecast_data['mean'],
                                    mode='lines',
                                    name='Intervention',
                                    line=dict(color='red')
                                ))
                                
                                # Uncertainty bands
                                fig.add_trace(go.Scatter(
                                    x=future_times,
                                    y=forecast_data['percentile_95'],
                                    fill=None,
                                    mode='lines',
                                    line_color='rgba(0,0,0,0)',
                                    showlegend=False
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=future_times,
                                    y=forecast_data['percentile_5'],
                                    fill='tonexty',
                                    mode='lines',
                                    line_color='rgba(0,0,0,0)',
                                    name='95% CI',
                                    fillcolor='rgba(255,0,0,0.2)'
                                ))
                                
                                fig.update_layout(
                                    title=f"Intervention Effect on {target_var}",
                                    xaxis_title="Time",
                                    yaxis_title=target_var,
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error running intervention simulation: {str(e)}")


def render_alerts_page(df, models):
    """Render the alerts and monitoring page."""
    st.markdown("# 🚨 Alerts & Monitoring")
    
    if df.empty:
        st.warning("No data available for monitoring")
        return
    
    # Change point detection
    st.markdown("## 📊 Change Point Detection")
    
    if st.button("Detect Change Points"):
        with st.spinner("Detecting structural breaks..."):
            try:
                change_detector = models['change_detector']
                
                # Detect change points
                change_points = change_detector.detect_change_points(df)
                
                if change_points:
                    # Display alerts for recent change points
                    alerts = change_detector.get_change_point_alerts(confidence_threshold=0.7)
                    
                    if alerts:
                        st.markdown("### 🚨 Recent Change Point Alerts")
                        for alert in alerts[:5]:  # Show top 5
                            alert_class = "alert-danger" if alert['confidence'] > 0.9 else "alert-warning"
                            st.markdown(
                                f'<div class="alert-box {alert_class}">'
                                f'<strong>{alert["variable"]}</strong> - '
                                f'Change detected at {alert["change_point_timestamp"]} '
                                f'(Confidence: {alert["confidence"]:.2f})</div>',
                                unsafe_allow_html=True
                            )
                    
                    # Visualize change points for selected variable
                    st.markdown("### 📈 Change Point Visualization")
                    
                    viz_variable = st.selectbox(
                        "Select Variable for Visualization",
                        list(change_points.keys())
                    )
                    
                    if viz_variable:
                        viz_data = change_detector.visualize_change_points(df, viz_variable)
                        
                        if viz_data:
                            fig = go.Figure()
                            
                            # Plot time series
                            fig.add_trace(go.Scatter(
                                x=viz_data['timestamps'],
                                y=viz_data['values'],
                                mode='lines',
                                name=viz_variable,
                                line=dict(color='blue')
                            ))
                            
                            # Add change points
                            for cp in viz_data['change_points']:
                                fig.add_vline(
                                    x=cp['timestamp'],
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"CP (conf: {cp['confidence']:.2f})"
                                )
                            
                            fig.update_layout(
                                title=f"Change Points in {viz_variable}",
                                xaxis_title="Time",
                                yaxis_title=viz_variable,
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("No significant change points detected")
                
            except Exception as e:
                st.error(f"Error detecting change points: {str(e)}")
    
    # Model performance monitoring
    st.markdown("## 📊 Model Performance Monitoring")
    
    if 'trained_model' in st.session_state:
        model = st.session_state['trained_model']
        
        # Check for model drift
        drift_results = model.detect_concept_drift()
        
        if drift_results:
            st.markdown("### 🔄 Model Drift Detection")
            
            drift_detected = any(drift_results.values())
            
            if drift_detected:
                st.markdown(
                    '<div class="alert-box alert-warning">'
                    '<strong>Model Drift Detected!</strong> Consider retraining the model.</div>',
                    unsafe_allow_html=True
                )
                
                # Show which metrics indicate drift
                for metric, is_drifting in drift_results.items():
                    status = "🔴 Drifting" if is_drifting else "🟢 Stable"
                    st.write(f"**{metric}**: {status}")
            else:
                st.markdown(
                    '<div class="alert-box alert-success">'
                    '<strong>Model Performance Stable</strong> - No significant drift detected.</div>',
                    unsafe_allow_html=True
                )


def main():
    """Main dashboard application."""
    try:
        # Load data and initialize models
        df, pipeline = load_data_cached()
        models = initialize_models()
        
        # Render sidebar
        page, model_type = render_sidebar()
        
        # Render selected page
        if page == "Overview":
            render_overview_page(df, models)
        elif page == "Data Sources":
            render_data_sources_page(df, pipeline)
        elif page == "Model Performance":
            render_model_performance_page(df, models, model_type)
        elif page == "Causal Analysis":
            render_causal_analysis_page(df, models)
        elif page == "What-If Scenarios":
            render_intervention_page(df, models)
        elif page == "Alerts & Monitoring":
            render_alerts_page(df, models)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Adaptive Socio-Economic Modeling Framework** | "
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
        logger.error(f"Dashboard error: {str(e)}")


def run_dashboard():
    """Entry point for running the dashboard."""
    main()


if __name__ == "__main__":
    run_dashboard()