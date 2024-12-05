import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Stock Market Seasonality Analyzer",
    page_icon="📈",
    layout="wide"
)

class StreamlitSeasonalityAnalyzer:
    def __init__(self, data, price_column='close'):
        """Initialize the analyzer with stock data"""
        self.data = data.copy()
        self.price_column = price_column
        
        # Calculate returns
        self.data['returns'] = self.data[price_column].pct_change()

    def decompose_series(self, period=252):
        """Decompose time series with plotly visualization"""
        decomposition = seasonal_decompose(
            self.data[self.price_column],
            period=period,
            extrapolate_trend='freq'
        )

        # Create figure with secondary y-axis
        fig = go.Figure()

        # Add traces
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data[self.price_column],
                      name="Original")
        )
        fig.add_trace(
            go.Scatter(x=self.data.index, y=decomposition.trend,
                      name="Trend")
        )
        fig.add_trace(
            go.Scatter(x=self.data.index, y=decomposition.seasonal,
                      name="Seasonal")
        )
        fig.add_trace(
            go.Scatter(x=self.data.index, y=decomposition.resid,
                      name="Residual")
        )

        fig.update_layout(
            title="Time Series Decomposition",
            xaxis_title="Date",
            yaxis_title="Price",
            height=800
        )

        return fig

    def plot_monthly_returns(self):
        """Create monthly returns box plot using plotly"""
        # Group by month and calculate mean returns
        monthly_returns = self.data['returns'].groupby(self.data.index.to_period('M')).mean()

        # Convert to DataFrame
        monthly_returns = monthly_returns.reset_index()
        monthly_returns.columns = ['period', 'returns']  # Rename columns for clarity

        # Extract year and month
        monthly_returns['year'] = monthly_returns['period'].dt.year
        monthly_returns['month'] = monthly_returns['period'].dt.month

        # Map month numbers to month names
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        monthly_returns['month_name'] = monthly_returns['month'].map(month_map)

        # Create box plot
        import plotly.express as px
        fig = px.box(
            monthly_returns,
            x='month_name',
            y='returns',
            title='Monthly Returns Distribution',
            labels={'returns': 'Returns', 'month_name': 'Month'}
        )

        return fig


    def plot_yearly_pattern(self):
        """Create heatmap of returns by month and year using plotly"""
        yearly_pattern = self.data['returns'].groupby(
            [self.data.index.year, self.data.index.month]
        ).mean().unstack()
        
        fig = go.Figure(data=go.Heatmap(
            z=yearly_pattern.values,
            x=[f"Month {i}" for i in range(1, 13)],
            y=yearly_pattern.index,
            colorscale='RdYlGn',
            text=np.round(yearly_pattern.values * 100, 2),
            texttemplate='%{text}%'
        ))
        
        fig.update_layout(
            title='Average Returns by Month and Year',
            xaxis_title='Month',
            yaxis_title='Year'
        )
        
        return fig

    def get_statistics(self):
        """Calculate various seasonality statistics"""
        # Monthly statistics
        monthly_groups = [group for _, group in self.data['returns'].groupby(self.data.index.month)]
        monthly_f_stat, monthly_p_value = stats.f_oneway(*monthly_groups)
        
        # Day of week statistics
        dow_returns = self.data['returns'].groupby(self.data.index.dayofweek).mean()
        dow_returns.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        dow_groups = [group for _, group in self.data['returns'].groupby(self.data.index.dayofweek)]
        dow_f_stat, dow_p_value = stats.f_oneway(*dow_groups)
        
        return {
            'monthly_stats': {
                'f_statistic': monthly_f_stat,
                'p_value': monthly_p_value
            },
            'dow_stats': {
                'f_statistic': dow_f_stat,
                'p_value': dow_p_value,
                'mean_returns': dow_returns
            }
        }

def main(data):
    
    # Initialize analyzer
    analyzer = StreamlitSeasonalityAnalyzer(data)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Time Series Decomposition",
        "Monthly Analysis",
        "Yearly Pattern",
        "Statistics"
    ])
    
    with tab1:
        st.plotly_chart(analyzer.decompose_series(), use_container_width=True)
    
    with tab2:
        st.plotly_chart(analyzer.plot_monthly_returns(), use_container_width=True)
    
    with tab3:
        st.plotly_chart(analyzer.plot_yearly_pattern(), use_container_width=True)
    
    with tab4:
        stats = analyzer.get_statistics()
        
        st.subheader("Monthly Effect")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F-statistic", f"{stats['monthly_stats']['f_statistic']:.4f}")
        with col2:
            st.metric("P-value", f"{stats['monthly_stats']['p_value']:.4f}")
        
        st.subheader("Day-of-Week Effect")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F-statistic", f"{stats['dow_stats']['f_statistic']:.4f}")
        with col2:
            st.metric("P-value", f"{stats['dow_stats']['p_value']:.4f}")
        
        st.subheader("Average Returns by Day of Week")
        st.dataframe(stats['dow_stats']['mean_returns'].round(6))

if __name__ == "__main__":
    main()