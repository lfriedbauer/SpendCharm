"""Spend Visibility Dashboard - Main Application
Cost optimization consulting tool using FinBERT
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Spend Visibility Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Your+Logo", width=150)
    st.markdown("---")
    
    # Date range selector
    date_range = st.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )
    
    # Category filter
    categories = ["All", "Cloud Services", "Software", "Marketing", "Office", "Travel", "Other"]
    selected_category = st.selectbox("Filter by Category", categories)
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    # Settings
    with st.expander("⚙️ Settings"):
        alert_threshold = st.slider("Alert Threshold (%)", 5, 50, 10)
        auto_refresh = st.checkbox("Auto-refresh (5 min)")

# Main content
st.markdown('<h1 class="main-header">💰 Spend Visibility Dashboard</h1>', unsafe_allow_html=True)

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Monthly Spend",
        value="$47,582",
        delta="↑ $3,247 (7.3%)",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label="Identified Savings",
        value="$8,947",
        delta="18.8% of spend",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="Active Vendors",
        value="73",
        delta="↑ 5 new this month"
    )

with col4:
    st.metric(
        label="Risk Alerts",
        value="4",
        delta="↓ 2 from last week",
        delta_color="normal"
    )

# Alert section
if st.session_state.alerts:
    st.markdown("### 🚨 Active Alerts")
    alert_container = st.container()
    with alert_container:
        for alert in st.session_state.alerts[:3]:  # Show top 3 alerts
            st.warning(f"⚠️ {alert}")

# Main dashboard tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Trends", "🏢 Vendors", "💡 Insights", "📄 Reports"])

with tab1:
    # Overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Spend by category donut chart
        st.subheader("Spend by Category")
        
        # Sample data
        category_data = pd.DataFrame({
            'Category': ['Cloud Services', 'Software', 'Marketing', 'Office', 'Travel', 'Other'],
            'Amount': [15000, 12000, 8000, 5000, 4582, 3000],
            'Percentage': [31.5, 25.2, 16.8, 10.5, 9.6, 6.3]
        })
        
        fig = px.pie(category_data, values='Amount', names='Category', hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top vendors bar chart
        st.subheader("Top 10 Vendors by Spend")
        
        vendor_data = pd.DataFrame({
            'Vendor': ['AWS', 'Microsoft', 'Google', 'Adobe', 'Slack', 
                      'Zoom', 'Salesforce', 'HubSpot', 'Mailchimp', 'Dropbox'],
            'Spend': [8500, 6200, 4800, 3200, 2800, 2500, 2300, 2100, 1900, 1700]
        })
        
        fig = px.bar(vendor_data, x='Spend', y='Vendor', orientation='h',
                    color='Spend', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Trends analysis
    st.subheader("Monthly Spend Trend")
    
    # Generate sample trend data
    months = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    trend_data = pd.DataFrame({
        'Month': months,
        'Actual': [42000, 43500, 44200, 45800, 44900, 46200, 47100, 46800, 47582, 48200, 49000, 50000],
        'Budget': [45000, 45000, 45000, 46000, 46000, 46000, 47000, 47000, 47000, 48000, 48000, 48000]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_data['Month'], y=trend_data['Actual'],
                             mode='lines+markers', name='Actual Spend',
                             line=dict(color='#1f77b4', width=3)))
    fig.add_trace(go.Scatter(x=trend_data['Month'], y=trend_data['Budget'],
                             mode='lines', name='Budget',
                             line=dict(color='#ff7f0e', width=2, dash='dash')))
    
    fig.update_layout(hovermode='x unified', xaxis_title="Month", yaxis_title="Spend ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly detection
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Anomalies")
        anomaly_data = pd.DataFrame({
            'Vendor': ['AWS', 'Adobe', 'Google Ads'],
            'Expected': [7500, 3000, 2000],
            'Actual': [8500, 3200, 2800],
            'Variance': ['+13.3%', '+6.7%', '+40%'],
            'Status': ['🟡 Review', '✅ OK', '🔴 Alert']
        })
        st.dataframe(anomaly_data, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Category Growth Rate")
        growth_data = pd.DataFrame({
            'Category': ['Cloud', 'Software', 'Marketing', 'Office'],
            'Growth': [12, -5, 8, -2]
        })
        fig = px.bar(growth_data, x='Category', y='Growth', 
                    color='Growth', color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Vendor management
    st.subheader("Vendor Analysis")
    
    # Search and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        vendor_search = st.text_input("🔍 Search vendors", placeholder="Type vendor name...")
    with col2:
        risk_filter = st.selectbox("Risk Level", ["All", "High", "Medium", "Low"])
    with col3:
        sort_by = st.selectbox("Sort by", ["Spend (High to Low)", "Name (A-Z)", "Risk Level"])
    
    # Vendor table
    vendor_details = pd.DataFrame({
        'Vendor': ['AWS', 'Microsoft', 'Google', 'Adobe', 'Slack', 'Zoom'],
        'Category': ['Cloud', 'Software', 'Cloud', 'Software', 'Software', 'Software'],
        'Monthly Spend': ['$8,500', '$6,200', '$4,800', '$3,200', '$2,800', '$2,500'],
        'YoY Change': ['+12%', '+5%', '+18%', '-3%', '+8%', '+15%'],
        'Risk': ['🟢 Low', '🟢 Low', '🟡 Medium', '🟢 Low', '🟢 Low', '🔴 High'],
        'Action': ['Optimize', 'Renew', 'Review', 'OK', 'OK', 'Renegotiate']
    })
    
    st.dataframe(
        vendor_details,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Vendor": st.column_config.TextColumn("Vendor", width="medium"),
            "Monthly Spend": st.column_config.TextColumn("Monthly Spend", width="small"),
        }
    )

with tab4:
    # AI Insights
    st.subheader("🤖 AI-Generated Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **💡 Cost Optimization Opportunities**
        
        1. **AWS Reserved Instances**: Switch 60% of on-demand instances to save ~$2,100/month
        2. **Software Consolidation**: Merge Slack + Teams → Save $1,400/month
        3. **Annual Contracts**: Convert 5 monthly subscriptions → Save $3,200/year
        4. **Unused Licenses**: 23 unused Adobe licenses detected → Save $780/month
        """)
    
    with col2:
        st.warning("""
        **⚠️ Risk Factors**
        
        • **Zoom**: 40% price increase detected - negotiate or switch
        • **Marketing spend**: Exceeded budget 3 months consecutively
        • **Vendor concentration**: 65% spend with top 3 vendors
        • **Contract renewals**: 4 contracts expire within 30 days
        """)
    
    # Sentiment Analysis
    st.subheader("Vendor Communication Sentiment")
    sentiment_data = pd.DataFrame({
        'Vendor': ['AWS', 'Microsoft', 'Adobe', 'Zoom', 'Slack'],
        'Positive': [85, 90, 70, 45, 88],
        'Neutral': [10, 8, 20, 30, 10],
        'Negative': [5, 2, 10, 25, 2]
    })
    
    fig = px.bar(sentiment_data, x='Vendor', y=['Positive', 'Neutral', 'Negative'],
                color_discrete_map={'Positive': '#2ca02c', 'Neutral': '#ff7f0e', 'Negative': '#d62728'})
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    # Reports section
    st.subheader("📄 Generate Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "Vendor Report", "Savings Report"]
        )
        
        report_period = st.selectbox(
            "Period",
            ["Last Week", "Last Month", "Last Quarter", "Year to Date"]
        )
        
        include_sections = st.multiselect(
            "Include Sections",
            ["Spend Overview", "Trends", "Anomalies", "Recommendations", "Vendor Details"],
            default=["Spend Overview", "Trends", "Recommendations"]
        )
    
    with col2:
        st.markdown("### Report Preview")
        st.text_area(
            "Preview",
            value="""EXECUTIVE SUMMARY - November 2024
            
Total Spend: $47,582 (↑ 7.3% MoM)
Identified Savings: $8,947 (18.8%)
Key Actions: 
- Renegotiate Zoom contract
- Implement AWS Reserved Instances
- Consolidate software licenses

Top Risk: Vendor concentration...""",
            height=200,
            disabled=True
        )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Generate Report", type="primary", use_container_width=True):
            st.success("Report generated successfully!")
    with col2:
        st.button("📧 Email Report", use_container_width=True)
    with col3:
        st.button("⬇️ Download PDF", use_container_width=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
with col2:
    st.caption("Data sources: Email, AWS, Invoices")
with col3:
    st.caption("Powered by FinBERT AI")

# Auto-refresh logic
if auto_refresh:
    import time
    time.sleep(300)  # 5 minutes
    st.rerun()