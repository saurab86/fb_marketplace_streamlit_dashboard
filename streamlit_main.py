
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Facebook Marketplace Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cards
st.markdown("""
    <style>
        div[data-testid="metric-container"] {
            background-color: #FFFFFF;
            border: 1px solid #DCDCDC;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        div[data-testid="metric-container"] > div {
            width: 100%;
        }
        div[data-testid="metric-container"] label {
            color: #0066CC;
        }
    </style>
""", unsafe_allow_html=True)

# Function to clean price data
def clean_price(price):
    try:
        if pd.isna(price) or price is None:
            return 0.0
        if isinstance(price, (int, float)):
            return float(price)
        cleaned = price.strip().replace('$', '').replace(',', '')
        return float(cleaned) if cleaned else 0.0
    except (ValueError, AttributeError):
        return 0.0

# Function to extract year from title
def extract_year(title):
    try:
        if pd.isna(title) or title is None:
            return None
        import re
        match = re.search(r'\b(19|20)\d{2}\b', title)
        if match:
            return float(match.group())
        return None
    except (ValueError, AttributeError):
        return None

# Function to calculate metrics for score cards
def calculate_metrics(df):
    metrics = {
        'Price Metrics': {
            'Average Price': f"${df['price_clean'].mean():,.2f}",
            'Highest Price': f"${df['price_clean'].max():,.2f}",
            'Lowest Price': f"${df['price_clean'].min():,.2f}",
            'Price Range': f"${df['price_clean'].max() - df['price_clean'].min():,.2f}"
        },
        'Listing Metrics': {
            'Total Listings': len(df),
            'Unique Makes': df['make'].nunique(),
            'Locations': df['state'].nunique(),
            'Average Rating': f"{df['seller_rating'].mean():.1f}⭐"
        },
        'Vehicle Metrics': {
            'Average Year': f"{df['year'].mean():.0f}",
            'Newest Vehicle': f"{int(df['year'].max())}" if df['year'].notna().any() else "N/A",
            'Oldest Vehicle': f"{int(df['year'].min())}" if df['year'].notna().any() else "N/A",
            'Age Range': f"{int(df['year'].max() - df['year'].min())} years" if df['year'].notna().any() else "N/A"
        },
        'Seller Metrics': {
            'Total Sellers': df['seller_name'].nunique(),
            'Top Rated Sellers': len(df[df['seller_rating'] >= 4]),
            'Unrated Sellers': len(df[df['seller_rating'] == 0]),
            'Average Response Time': 'N/A'  # Placeholder for future data
        }
    }
    return metrics

# Function to load and prepare data
@st.cache_data
def load_data():
    df = pd.read_json('output.json')
    df['price_clean'] = df['price'].apply(clean_price)
    df['year'] = df['title'].apply(extract_year)
    df['make'] = df['title'].str.split().str[1:2].str[0]
    df['state'] = df['location'].str.split(', ').str[-1]
    
    return df

# Function to display score cards
def display_score_cards(metrics):
    st.markdown("### 📊 Marketplace Analytics Score Cards")
    
    # Create tabs for different metric categories
    tabs = st.tabs(list(metrics.keys()))
    
    for tab, (category, category_metrics) in zip(tabs, metrics.items()):
        with tab:
            cols = st.columns(len(category_metrics))
            for col, (metric_name, value) in zip(cols, category_metrics.items()):
                with col:
                    st.metric(
                        label=metric_name,
                        value=value
                    )

# Main app
def main():
    st.title("Facebook Marketplace Data Analysis")
    
    try:
        df = load_data()
        
        # Calculate metrics for score cards
        metrics = calculate_metrics(df)
        
        # Display score cards
        display_score_cards(metrics)
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        min_price = float(df['price_clean'].min())
        max_price = float(df['price_clean'].max())
        price_range = st.sidebar.slider(
            "Price Range",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )
        
        valid_years = df['year'].dropna()
        if not valid_years.empty:
            min_year = int(valid_years.min())
            max_year = int(valid_years.max())
            year_range = st.sidebar.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            
            mask = (
                (df['price_clean'].between(price_range[0], price_range[1])) &
                (df['year'].between(year_range[0], year_range[1]))
            )
            filtered_df = df[mask]
        else:
            filtered_df = df[df['price_clean'].between(price_range[0], price_range[1])]
        
        # Data visualizations
        st.markdown("### 📈 Market Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if not filtered_df.empty:
                fig_price = px.box(filtered_df, x='make', y='price_clean',
                                 title="Price Distribution by Make")
                st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.warning("No data available for price distribution chart")
        
        with col2:
            if not filtered_df.empty and 'year' in filtered_df.columns:
                avg_price_year = filtered_df.groupby('year')['price_clean'].mean().reset_index()
                fig_year = px.line(avg_price_year, x='year', y='price_clean',
                                 title="Average Price Trend by Year")
                st.plotly_chart(fig_year, use_container_width=True)
            else:
                st.warning("No data available for price trend chart")
        
        # Location Analysis
        st.markdown("### 📍 Geographic Distribution")
        if not filtered_df.empty:
            location_counts = filtered_df['state'].value_counts()
            fig_location = px.bar(location_counts, title="Number of Listings by State")
            st.plotly_chart(fig_location, use_container_width=True)
        else:
            st.warning("No location data available")
        
        # Listings table
        st.markdown("### 📋 Detailed Listings")
        if not filtered_df.empty:
            st.dataframe(
                filtered_df[['title', 'price', 'location', 'year', 'make', 'seller_rating']],
                use_container_width=True
            )
        else:
            st.warning("No listings match the selected filters")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
