import re
import os
import boto3
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from streamlit_date_picker import date_range_picker, date_picker, PickerType

class FacebookMarketplaceAnalysis:
    def __init__(self):
        st.set_page_config(
            page_title='Facebook Marketplace Analysis',
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("Facebook Market Data Analysis")

        self.df = self.load_data_from_dynamoDB()
        self.df['price'] = self.df['price'].apply(self.clean_price)
        self.df['resale_value'] = self.df['resale_value'].apply(self.clean_price)

    ### Load Data from dynamoDB ###
    def load_data_from_dynamoDB(self):
        dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=os.getenv("aws_access_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_access_key"),
            region_name="us-east-1"
        )

        location_table = dynamodb.Table("location")
        listing_table = dynamodb.Table("listing")

        response = listing_table.scan()
        listing_data = response.get("Items", [])
        df = pd.DataFrame(listing_data)
        df = self.add_filters_and_search(df)
        return df

    ### Filters in Sidebar ###
    def add_filters_and_search(self, df):
        st.sidebar.title("Filters & Search")

        # Date range picker
        default_start, default_end = datetime(2025, 1, 1), datetime.now()
        refresh_value = timedelta(days=1)
        refresh_buttons = [{'button_name': 'Reset', 'refresh_value': refresh_value}]

        with st.sidebar:
            st.markdown("#### Select Date Range")
            date_range_string = date_range_picker(
                picker_type=PickerType.date,
                start=default_start,
                end=default_end,
                refresh_buttons=refresh_buttons,
                key='date_range_picker'
            )

        if date_range_string:
            start_date, end_date = date_range_string
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        else:
            start_date, end_date = default_start, default_end

        # Keyword Search
        search_term = st.sidebar.text_input("Search Listings", "")
        categories = ["All"] + sorted(df["category"].unique().tolist())
        selected_category = st.sidebar.selectbox("Select Category", categories)

        #Score Filter
        score_ranges = {
            "All": (0, 100),
            "High (80-100)": (80, 100),
            "Medium (50-79)": (50, 79),
            "Low (0-49)": (0, 49)
        }
        selected_score_range = st.sidebar.selectbox("Filter by Score", list(score_ranges.keys()))

        #Sort Options
        sort_options = {
            "Profitability (High to Low)": ("profit_margin", False),
            "Demand Score (High to Low)": ("score", False),
            "Most Recent First": ("timestamp", False),
            "Price (High to Low)": ("price", False),
            "Price (Low to High)": ("price", True)
        }
        selected_sort = st.sidebar.selectbox("Sort By", list(sort_options.keys()))

        #Apply Filters
        filtered_df = df.copy()

            #Apply Date Range filter
        if "timestamp" in filtered_df.columns:
            filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"], errors="coerce")
            filtered_df = filtered_df[filtered_df["timestamp"].notna()]
            filtered_df = filtered_df[(filtered_df["timestamp"] >= start_date) & (filtered_df["timestamp"] <= end_date)]

        if search_term:
            filtered_df = filtered_df[
                filtered_df["title"].str.contains(search_term, case=False, na=False) |
                filtered_df["description"].str.contains(search_term, case=False, na=False)
            ]

            #Apply Keyword search if provided
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["category"] == selected_category]

            #Apply Score filter
        if selected_score_range != "All":
            min_score, max_score = score_ranges[selected_score_range]
            filtered_df = filtered_df[(filtered_df["score"] >= min_score) & (filtered_df["score"] <= max_score)]

            #Apply sorting
        sort_column, ascending = sort_options[selected_sort]
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

        st.sidebar.divider()
        st.sidebar.subheader("Filter Summary")
        st.sidebar.write(f"Showing {len(filtered_df)} of {len(df)} listings")

        return filtered_df

    ### Clean price column ###
    def clean_price(self, price_str):
        """
        Clean price strings by extracting numeric values and handling various price formats.
        
        Args:
            price_str: String containing price information (e.g., "$1,234.56", "1234", "$1k", "1.2K", etc.)
            
        Returns:
            float: Cleaned price value or np.nan if no valid price found
        """
        if pd.isna(price_str):
            return np.nan

        price_str = str(price_str).strip()
        price_str = re.sub(r'[$,\s]', '', price_str)

        if 'k' in price_str.lower():
            try:
                price_str = str(float(price_str.lower().replace('k', '')) * 1000)
            except ValueError:
                return np.nan

        if '-' in price_str or ' to ' in price_str:
            price_str = re.split(r'[-\s]+to[\s]+|[-]', price_str)[0]

        matches = re.findall(r'\d*\.?\d+', price_str)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                return np.nan

        return np.nan

    ### Overview KPI ###
    def overview_metrics_kpi(self):
        #Overview KPI
        st.subheader("Overview Metrics:")
        if self.df.empty:
            st.warning("No data available after applying filters.")
            return

        total_listings = len(self.df)
        category_counts = self.df["category"].value_counts()

        resale_value_median = self.df['resale_value'].replace(0, pd.NA).median()
        self.df['resale_value'] = self.df['resale_value'].fillna(resale_value_median)

        price_median = self.df['price'].replace(0, pd.NA).median()
        self.df['price'] = self.df['price'].fillna(price_median)
        self.df['price'] = self.df['price'].replace(0, price_median)

        total_resale_value = self.df['resale_value'].sum()
        total_price = self.df['price'].sum()

        if total_price == 0:
            avg_profit_margin = 0
        else:
            avg_profit_margin = ((total_resale_value - total_price) / total_price) * 100

        high_scoring_listings = len(self.df[self.df["score"] > 80])
        self.df["recency_weight"] = self.df["recency_weight"].astype(float)
        high_demand_listings = len(self.df[self.df["recency_weight"] > self.df["recency_weight"].quantile(0.75)])

        self.df["seller_rating"] = pd.to_numeric(self.df["seller_rating"], errors='coerce')
        self.df = self.df.dropna(subset=["seller_rating"])
        low_competition_listings = len(self.df[self.df["seller_rating"] > self.df["seller_rating"].quantile(0.75)])

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("### Total Listings", total_listings, border=1)

        with col2:
            st.metric("Avg Profit Margin", f"${avg_profit_margin:.2f}", border=1)

        with col3:
            st.metric("High-Scoring Listings(>80)", high_scoring_listings, border=1)

        with col4:
            st.metric("High-Demand Listings", high_demand_listings, border=1)

        with col5:
            st.metric("Low-Competition Listings", low_competition_listings, border=1)

        st.write("")
        st.subheader("Additional Insights:")
        st.write("#### Category Distribution")
        st.bar_chart(category_counts, x_label="Categories", y_label="Listing Count")
        st.divider()

    ### Charts and Graphs ###
    def charts_and_graphs(self):
        if self.df.empty:
            st.warning("No data available after applying filters.")
            return

        self.price_distribution_by_category()
        self.profit_margin_vs_listing_price()
        self.top_5_most_profitable_categories()
        self.listing_recency_heatmap()

    def price_distribution_by_category(self):
        st.subheader("Price Distribution by Category")
        viz_type = st.radio("Select Visualization Type", ["Histogram", "Box Plot"], key="viz_type")

        if viz_type == "Histogram":
            df_filtered = self.df[self.df['price'] <= 200000]
            fig = px.histogram(
                df_filtered,
                x="price",
                color="category",
                nbins=30,
                labels={'price': 'Price ($)', 'count': 'Number of Listings'},
                opacity=0.9
            )
            fig.update_layout(
                barmode='overlay',
                xaxis_title="Price ($)",
                yaxis_title="Number of Listings",
                legend_title="Category",
                height=500,
                yaxis_type="log"
            )
        else:
            fig = px.box(
                self.df,
                x="category",
                y="price",
                labels={'price': 'Price ($)', 'category': 'Category'},
                points="outliers"
            )
            fig.update_layout(
                xaxis_title="Category",
                yaxis_title="Price ($)",
                height=500,
                xaxis={'tickangle': 45}
            )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Price Summary Statistics by Category")
        price_stats = self.df.groupby('category')['price'].agg([
            'count',
            'mean',
            'median',
            'std',
            'min',
            'max'
        ]).round(2)

        st.dataframe(price_stats, use_container_width=True)
        st.divider()

    def profit_margin_vs_listing_price(self):
        st.write("")
        category_profits = self.df.groupby('category').agg({
            'profit_margin': ['mean', 'count', 'std'],
            'price': 'mean'
        }).round(2)

        category_profits.columns = ['avg_profit', 'listing_count', 'profit_std', 'avg_price']
        category_profits = category_profits.reset_index()
        category_profits = category_profits.sort_values('avg_profit', ascending=True)

        fig = px.bar(
            category_profits,
            y='category',
            x='avg_profit',
            orientation='h',
            labels={'category': 'Category', 'avg_profit': 'Average Profit Margin ($)'},
            hover_data={'listing_count': True, 'profit_std': ':.2f', 'avg_price': ':.2f'},
            color='avg_profit',
            color_continuous_scale='RdYlGn'
        )

        fig.update_layout(
            height=600,
            xaxis_title="Average Profit Margin ($)",
            yaxis_title="Category",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis=dict(showgrid=False)
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            most_profitable = category_profits.iloc[-1]
            st.metric("Most Profitable Category", most_profitable['category'], f"${most_profitable['avg_profit']:.2f} avg margin")

        with col2:
            total_categories = len(category_profits)
            st.metric("Total Categories", total_categories, f"{category_profits['listing_count'].sum()} total listings")

        with col3:
            overall_avg = self.df['profit_margin'].mean()
            st.metric("Overall Average Margin", f"${overall_avg:.2f}")

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Detailed Category Statistics"):
            category_profits['profit_cv'] = (category_profits['profit_std'] / category_profits['avg_profit']).round(3)
            display_stats = category_profits[[
                'category', 'avg_profit', 'profit_std', 'profit_cv', 'listing_count', 'avg_price'
            ]].rename(columns={
                'avg_profit': 'Avg Profit ($)',
                'profit_std': 'Std Dev ($)',
                'profit_cv': 'Variability',
                'listing_count': 'Listings',
                'avg_price': 'Avg Price ($)',
                'category': 'Category'
            })
            st.dataframe(display_stats, use_container_width=True)
        st.divider()

    def top_5_most_profitable_categories(self):
        st.subheader("Top 5 Most Profitable Categories")
        category_profits = self.df.groupby('category').agg({
            'profit_margin': ['mean', 'count', 'std'],
            'price': 'mean'
        }).round(2)

        category_profits.columns = ['avg_profit', 'listing_count', 'profit_std', 'avg_price']
        category_profits = category_profits.reset_index()
        category_profits['avg_profit'] = pd.to_numeric(category_profits['avg_profit'], errors='coerce')
        category_profits = category_profits.dropna(subset=['avg_profit'])
        top_5_categories = category_profits.nlargest(5, 'avg_profit')

        fig = px.bar(
            top_5_categories,
            x='category',
            y='avg_profit',
            labels={'category': 'Category', 'avg_profit': 'Average Profit Margin ($)'},
            hover_data={'listing_count': True, 'profit_std': ':.2f', 'avg_price': ':.2f'},
            color='avg_profit',
            color_continuous_scale='Greens'
        )

        fig.update_layout(
            height=500,
            xaxis_title="Category",
            yaxis_title="Average Profit Margin ($)",
            xaxis_tickangle=-45,
            showlegend=False,
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
        )

        st.plotly_chart(fig, use_container_width=True)
        st.divider()

    def listing_recency_heatmap(self):
        st.subheader("Listing Recency Heatmap")
        self.df['timestamp'] = self.df['timestamp'].replace("N/A", pd.NaT)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        self.df = self.df.dropna(subset=['timestamp'])
        self.df['hour'] = self.df['timestamp'].dt.hour

        def safe_int_convert(x):
            try:
                if pd.isna(x) or x == "N/A":
                    return 0
                return int(float(x))
            except (ValueError, TypeError):
                return 0

        def format_hour(hour):
            try:
                hour = safe_int_convert(hour)
                if hour < 0 or hour > 23:
                    return "Invalid Time"
                time_obj = datetime.strptime(f"{hour:02d}:00", "%H:%M")
                return time_obj.strftime("%I:%M %p")
            except:
                return "Unknown"

        hourly_counts = self.df.groupby(self.df['hour'].apply(safe_int_convert)).size().reset_index(name='count')
        hourly_counts['time_label'] = hourly_counts['hour'].apply(format_hour)

        fig = px.line(
            hourly_counts,
            x='hour',
            y='count',
            title='Listing Activity Throughout the Day',
            labels={'hour': 'Time of Day', 'count': 'Number of Listings'}
        )

        fig.update_layout(
            height=500,
            xaxis=dict(
                tickmode='array',
                ticktext=hourly_counts['time_label'],
                tickvals=hourly_counts['hour'],
                tickangle=45,
                title="Time of Day",
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            ),
            yaxis=dict(
                title="Number of Listings",
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            )
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            peak_hour = hourly_counts.loc[hourly_counts['count'].idxmax()]
            st.metric("Peak Activity Time", peak_hour['time_label'], f"{peak_hour['count']} listings")

        with col2:
            quiet_hour = hourly_counts.loc[hourly_counts['count'].idxmin()]
            st.metric("Lowest Activity Time", quiet_hour['time_label'], f"{quiet_hour['count']} listings")

        with col3:
            avg_listings = hourly_counts['count'].mean()
            st.metric("Average Listings per Hour", f"{avg_listings:.1f}")

        st.plotly_chart(fig, use_container_width=True)

    def start_streamlit_app(self):
        self.overview_metrics_kpi()
        self.charts_and_graphs()

if __name__ == '__main__':
    analysis = FacebookMarketplaceAnalysis()
    analysis.start_streamlit_app()