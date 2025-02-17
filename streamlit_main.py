import re
import boto3
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.figure_factory as ff


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


### Load Data from dynamo DB
def load_data_from_dynamoDB():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")

    # DynamoDB Tables
    location_table = dynamodb.Table("location")
    listing_table = dynamodb.Table("listing")

    response = listing_table.scan()
    listing_data = response.get("Items", [])
    df = pd.DataFrame(listing_data)
    df=add_filters_and_search(df)
    return df


def add_filters_and_search(df):
    st.sidebar.title("Filters & Search")
    
    # Keyword Search
    search_term = st.sidebar.text_input("Search Listings", "")
    
    # Category Filter
    categories = ["All"] + sorted(df["category"].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    # Score Filter
    score_ranges = {
        "All": (0, 100),
        "High (80-100)": (80, 100),
        "Medium (50-79)": (50, 79),
        "Low (0-49)": (0, 49)
    }
    selected_score_range = st.sidebar.selectbox("Filter by Score", list(score_ranges.keys()))
    
    # Sort Options
    sort_options = {
        "Profitability (High to Low)": ("profit_margin", False),
        "Demand Score (High to Low)": ("score", False),
        "Most Recent First": ("timestamp", False),
        "Price (High to Low)": ("price", False),
        "Price (Low to High)": ("price", True)
    }
    selected_sort = st.sidebar.selectbox("Sort By", list(sort_options.keys()))
    
    # Apply Filters
    filtered_df = df.copy()
    
    # Apply keyword search if provided
    if search_term:
        # Search in title and description (assuming these columns exist)
        filtered_df = filtered_df[
            filtered_df["title"].str.contains(search_term, case=False, na=False) |
            filtered_df["description"].str.contains(search_term, case=False, na=False)
        ]
    
    # Apply category filter
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]
    
    # Apply score filter
    if selected_score_range != "All":
        min_score, max_score = score_ranges[selected_score_range]
        filtered_df = filtered_df[
            (filtered_df["score"] >= min_score) & 
            (filtered_df["score"] <= max_score)
        ]
    
    # Apply sorting
    sort_column, ascending = sort_options[selected_sort]
    filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)
    
    # Display filter summary
    st.sidebar.divider()
    st.sidebar.subheader("Filter Summary")
    st.sidebar.write(f"Showing {len(filtered_df)} of {len(df)} listings")
    
    return filtered_df



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

    # Convert 'seller_rating' to numeric, coercing errors to NaN
    df["seller_rating"] = pd.to_numeric(df["seller_rating"], errors='coerce')

    # Optionally, you can drop rows with NaN values in 'seller_rating' if needed
    df = df.dropna(subset=["seller_rating"])

    # Now you can calculate the low_competition_listings without the error
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
    # df.to_csv("cleaned_df.csv")
    
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
    # Calculate average profit margin by category
    category_profits = df.groupby('category').agg({
        'profit_margin': ['mean', 'count', 'std'],
        'price': 'mean'
    }).round(2)
    
    # Flatten column names
    category_profits.columns = ['avg_profit', 'listing_count', 'profit_std', 'avg_price']
    category_profits = category_profits.reset_index()
    
    # Sort by average profit margin in descending order
    category_profits = category_profits.sort_values('avg_profit', ascending=True)
    
    # Create bar chart
    fig = px.bar(
        category_profits,
        y='category',
        x='avg_profit',
        orientation='h',  # Horizontal bars for better category label readability
        title='Average Profit Margin by Category',
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
    
    # Update layout
    fig.update_layout(
        height=600,
        xaxis_title="Average Profit Margin ($)",
        yaxis_title="Category",
        # Add gridlines
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        # Remove y-axis gridlines for cleaner look with bars
        yaxis=dict(
            showgrid=False
        )
    )
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_profitable = category_profits.iloc[-1]  # Last row after sorting
        st.metric(
            "Most Profitable Category",
            most_profitable['category'],
            f"${most_profitable['avg_profit']:.2f} avg margin"
        )
    
    with col2:
        total_categories = len(category_profits)
        st.metric(
            "Total Categories",
            total_categories,
            f"{category_profits['listing_count'].sum()} total listings"
        )
    
    with col3:
        overall_avg = df['profit_margin'].mean()
        st.metric(
            "Overall Average Margin",
            f"${overall_avg:.2f}"
        )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed statistics in an expander
    with st.expander("Detailed Category Statistics"):
        # Add profit margin variability (coefficient of variation)
        category_profits['profit_cv'] = (category_profits['profit_std'] / category_profits['avg_profit']).round(3)
        
        # Reorder columns for display
        display_stats = category_profits[[
            'category', 'avg_profit', 'profit_std', 'profit_cv', 
            'listing_count', 'avg_price'
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
    st.divider()

    ##### D.Listing Recency Heatmap #####
    st.subheader("Listing Recency Heatmap")
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # # Extract hour and convert to integer
    # df['hour'] = df['timestamp'].dt.hour
    
   # Safely convert to integer
    df['timestamp'] = df['timestamp'].replace("N/A", pd.NaT)

    # Convert the 'timestamp' column to datetime, coercing errors to NaT
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows where 'timestamp' is NaT (invalid or missing timestamps)
    df = df.dropna(subset=['timestamp'])

    # Extract hour and convert to integer
    df['hour'] = df['timestamp'].dt.hour

    # Safely convert to integer
    def safe_int_convert(x):
        try:
            if pd.isna(x) or x == "N/A":  # Handle NaN and "N/A" explicitly
                return 0
            return int(float(x))
        except (ValueError, TypeError):
            return 0

    # Format hour to readable time
    def format_hour(hour):
        try:
            hour = safe_int_convert(hour)
            if hour < 0 or hour > 23:  # Validate hour range
                return "Invalid Time"
            time_obj = datetime.strptime(f"{hour:02d}:00", "%H:%M")
            return time_obj.strftime("%I:%M %p")
        except:
            return "Unknown"

    # Count listings by hour
    hourly_counts = df.groupby(df['hour'].apply(safe_int_convert)).size().reset_index(name='count')
    hourly_counts['time_label'] = hourly_counts['hour'].apply(format_hour)

    # Create the line plot
    fig = px.line(
        hourly_counts,
        x='hour',
        y='count',
        title='Listing Activity Throughout the Day',
        labels={
            'hour': 'Time of Day',
            'count': 'Number of Listings'
        }
    )

    # Update layout for better visualization
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

    # Display the plot in Streamlit
    # st.plotly_chart(fig)

    
    # Add metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        peak_hour = hourly_counts.loc[hourly_counts['count'].idxmax()]
        st.metric(
            "Peak Activity Time",
            peak_hour['time_label'],
            f"{peak_hour['count']} listings"
        )
    
    with col2:
        quiet_hour = hourly_counts.loc[hourly_counts['count'].idxmin()]
        st.metric(
            "Lowest Activity Time",
            quiet_hour['time_label'],
            f"{quiet_hour['count']} listings"
        )
    
    with col3:
        avg_listings = hourly_counts['count'].mean()
        st.metric(
            "Average Listings per Hour",
            f"{avg_listings:.1f}"
        )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def start_streamlit_app():
    df=load_data_from_dynamoDB()
    overview_metrics_kpi(df)
    charts_and_graphs(df)


if __name__ == '__main__':
    start_streamlit_app()


