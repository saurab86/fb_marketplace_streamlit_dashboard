import re
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
from datetime import datetime
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

df = pd.read_csv("listing.csv")

# listing_data = listing_table.scan().get('Items')

st.set_page_config(
    page_title = ' Facebook Marketplace Analysis',
    page_icon = "📊",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

#Title
st.title("Facebook Market Data Analysis")

#Overview KPI
st.subheader("Overview Metrics:")


### Cleaning data
def clean_price(price_str):
    """
    Clean price strings by extracting numeric values and handling various price formats.
    
    Args:
        price_str: String containing price information (e.g., "$1,234.56", "1234", "$1k", "1.2K", etc.)
        
    Returns:
        float: Cleaned price value or np.nan if no valid price found
    """
    if pd.isna(price_str):
        return np.nan
    
    # Convert to string if not already
    price_str = str(price_str).strip()
    
    # Remove currency symbols, commas and spaces
    price_str = re.sub(r'[$,\s]', '', price_str)
    
    # Handle 'k' or 'K' notation (e.g., "1.5k" or "1.5K" = 1500)
    if 'k' in price_str.lower():
        try:
            # Remove 'k' or 'K' and multiply by 1000
            price_str = str(float(price_str.lower().replace('k', '')) * 1000)
        except ValueError:
            return np.nan
    
    # Handle ranges (e.g., "100-200", "100 to 200") - take the lower value
    if '-' in price_str or ' to ' in price_str:
        price_str = re.split(r'[-\s]+to[\s]+|[-]', price_str)[0]
    
    # Extract numeric values (including decimals)
    matches = re.findall(r'\d*\.?\d+', price_str)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return np.nan
            
    return np.nan

############# 1. Overview KPI ##########
def overview_metrics_kpi(df):
    total_listings = len(df)
    category_counts = df["category"].value_counts()
    avg_profit_margin = df["profit_margin"].mean()
    high_scoring_listings = len(df[df["score"] > 80])


    df["recency_weight"] = df["recency_weight"].astype(float)
    high_demand_listings = len(df[df["recency_weight"] > df["recency_weight"].quantile(0.75)])


    # Convert 'seller_rating' to numeric
    df["seller_rating"] = pd.to_numeric(df["seller_rating"], errors='coerce')

    df = df.dropna(subset=["seller_rating"])

    low_competition_listings = len(df[df["seller_rating"] > df["seller_rating"].quantile(0.75)])


    # KPI CARDS
    col1, col2, col3, col4, col5 = st.columns(5)
    # Total Listings
    with col1:
        st.metric("### Total Listings", total_listings,border=1)

    # Average Profit Margin
    with col2:
        st.metric("Avg Profit Margin", f"${avg_profit_margin:.2f}",border=1)

    # High-Scoring Listings
    with col3:
        st.metric("High-Scoring Listings(>80)", high_scoring_listings, border=1)

    # High-Demand Listings
    with col4:
        st.metric("High-Demand Listings", high_demand_listings,border=1)

    # Low-Competition Listings
    with col5:
        st.metric("Low-Competition Listings", low_competition_listings,border=1)

    # Additional Visualizations (Optional)
    st.write("")
    st.subheader("Additional Insights:")

    # Category Distribution
    st.write("#### Category Distribution")
    st.bar_chart(category_counts,x_label="Categories", y_label="Listing Count")

    st.divider()


################ 2. Charts and Graphs #############
def charts_and_graphs(df): 
    
    ##### A. Price Distribution by Category #####
    st.subheader("Price Distribution by Category")
    
    viz_type = st.radio(
        "Select Visualization Type",
        ["Histogram", "Box Plot"],
        key="viz_type"
    )
    
    # Convert price to numeric, handling any non-numeric values
    df['price'] = df['price'].apply(clean_price)
    df.to_csv("cleaned_df.csv")
    
    if viz_type == "Histogram":

        # Creating histogram using plotly
        fig = px.histogram(
            df,
            x="price",
            color="category",
            nbins=30,
            # title="Price Distribution by Category",
            labels={'price': 'Price ($)', 'count': 'Number of Listings'},
            opacity=0.9
        )
        
        # Updating layout for better readability
        fig.update_layout(
            barmode='overlay',
            xaxis_title="Price ($)",
            yaxis_title="Number of Listings",
            legend_title="Category",
            height=500,
        )
        
    else:  # Box Plot
        # creating box plot using plotly
        fig = px.box(
            df,
            x="category",
            y="price",
            title="Price Distribution by Category",
            labels={'price': 'Price ($)', 'category': 'Category'},
            points="outliers"
        )
        
        # updating layout for better readability
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Price ($)",
            height=500,
            xaxis={'tickangle': 45}
        )
    
    # displaying the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    st.subheader("Price Summary Statistics by Category")
    price_stats = df.groupby('category')['price'].agg([
        'count',
        'mean',
        'median',
        'std',
        'min',
        'max'
    ]).round(2)
    
    st.dataframe(price_stats, use_container_width=True)
    st.divider()

    ##### B.Profit Margin vs. Listing Price #####
    st.write("")
    st.subheader("Profit Margin Vs. Listing Price Analysis")

    correlation = df['price'].corr(df['profit_margin'])
    df['score'] = pd.to_numeric(df['score'], errors='coerce')

    # Calculating avg. profit margin by demand score quartiles
    df['demand_quartile'] = pd.qcut(df['score'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    avg_margin_by_demand = df.groupby('demand_quartile')['profit_margin'].mean()
    
    # Calculating optimal price ranges (where profit margins are highest)
    df['price_bracket'] = pd.qcut(df['price'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    optimal_price_range = df.groupby('price_bracket')['profit_margin'].mean().idxmax()
    
    # Creating color scale based on demand score
    fig = px.scatter(
        df,
        x='price',
        y='profit_margin',
        color='score',
        color_continuous_scale=['red', 'yellow', 'green'],
        labels={
            'price': 'Listing Price ($)',
            'profit_margin': 'Profit Margin ($)',
            'score': 'Demand Score'
        },
        hover_data=['category']  # Show category on hover
    )

    fig.update_layout(
        height=600,
        width=800,
        xaxis_title="Listing Price ($)",
        yaxis_title="Profit Margin ($)",
        coloraxis_colorbar_title="Demand Score",
        showlegend=True,
        
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
    )

    #correlation coffecient
    st.plotly_chart(fig, use_container_width=True)
    st.divider()


    ##### C. Top 5 Most Profitable Categories #####
    st.subheader("Average Profit Margin by Category")
    
    # Calculate average profit margin by category
    category_profits = df.groupby('category').agg({
        'profit_margin': ['mean', 'count', 'std'],
        'price': 'mean'
    }).round(2)
    
    # Flattening column names
    category_profits.columns = ['avg_profit', 'listing_count', 'profit_std', 'avg_price']
    category_profits = category_profits.reset_index()
    
    # Sort by average profit margin in descending order
    category_profits = category_profits.sort_values('avg_profit', ascending=True)
    
    # Creating bar chart
    fig = px.bar(
        category_profits,
        y='category',
        x='avg_profit',
        orientation='h',  # Horizontal bars for better category label readability
        labels={
            'category': 'Category',
            'avg_profit': 'Average Profit Margin ($)'
        },
        # Add hover data
        hover_data={
            'listing_count': True,  
            'profit_std': ':.2f',   
            'avg_price': ':.2f'     
        },

        # Color bars by profit margin
        color='avg_profit',
        color_continuous_scale='RdYlGn'  # Red to Yellow to Green color scale
    )

    fig.update_layout(
        height=600,
        xaxis_title="Average Profit Margin ($)",
        yaxis_title="Category",
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=False
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    #********************************************************************************************************
    st.subheader("Top 5 Most Profitable Categories")
    category_profits = df.groupby('category').agg({
        'profit_margin': ['mean', 'count', 'std'],
        'price': 'mean'
    }).round(2)
    
    # Flatten column names
    category_profits.columns = ['avg_profit', 'listing_count', 'profit_std', 'avg_price']
    category_profits = category_profits.reset_index()
    
    # Convert avg_profit to numeric, handling any non-numeric values
    category_profits['avg_profit'] = pd.to_numeric(category_profits['avg_profit'], errors='coerce')
    
    # Drop any rows with NaN values after conversion
    category_profits = category_profits.dropna(subset=['avg_profit'])
    
    # Sort by average profit margin and get top 5
    top_5_categories = category_profits.nlargest(5, 'avg_profit')
    
    # Create bar chart
    fig = px.bar(
        top_5_categories,
        x='category',
        y='avg_profit',
        labels={
            'category': 'Category',
            'avg_profit': 'Average Profit Margin ($)'
        },
        # Add hover data
        hover_data={
            'listing_count': True,    # Show number of listings
            'profit_std': ':.2f',     # Show standard deviation
            'avg_price': ':.2f'       # Show average price
        },
        # Color bars by profit margin
        color='avg_profit',
        color_continuous_scale='Greens'  # Use green color scale for profitability
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        xaxis_title="Category",
        yaxis_title="Average Profit Margin ($)",
        xaxis_tickangle=-45,  # Angle category labels for better readability
        showlegend=False,
        # Add gridlines
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    ##### D.Listing Recency Heatmap #####
    st.subheader("Listing Recency Heatmap")
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # # Extract hour and convert to integer
    # df['hour'] = df['timestamp'].dt.hour.fillna(0).astype(int)

    # time_series_data = df.groupby("hour").size().reset_index(name="count")
    # fig, ax = plt.subplots(figsize=(10, 5))
    # sns.heatmap(time_series_data.pivot(index="count", columns="hour", values="count"), cmap="Blues", annot=True, fmt="d", ax=ax)
    # st.pyplot(fig)

def start_streamlit_app(df):
    # overview_metrics_kpi(df)
    charts_and_graphs(df)


start_streamlit_app(df)
