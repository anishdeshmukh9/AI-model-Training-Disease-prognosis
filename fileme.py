import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time
from datetime import datetime, timedelta
import json
import os

# Set the page configuration
st.set_page_config(page_title="conlive", layout="wide")

# Initialize session state for refresh tracking and data caching
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = None
if 'api_calls_remaining' not in st.session_state:
    st.session_state.api_calls_remaining = 10  # Assumes 10 calls initially
if 'last_api_reset' not in st.session_state:
    st.session_state.last_api_reset = datetime.now()

# Display the title and caption
st.title("Crypto Live ")
st.caption("⏱️ Auto-refreshes data with rate limiting protection | Powered by CoinGecko API")

# Add sidebar for settings
with st.sidebar:
    st.header("Dashboard Settings")
    num_cryptos = st.slider("Number of cryptocurrencies", min_value=5, max_value=20, value=10)
    refresh_interval = st.slider("Refresh interval (seconds)", min_value=30, max_value=300, value=60)
    chart_type = st.selectbox("Price Chart Type", ["Bar", "Line"])
    
    # Use local cache option to avoid API calls
    use_local_cache = st.checkbox("Enable Fallback Mode", value=False, 
                                help="Use cached data when API limits are reached")
    
    # Show API status
    st.subheader("API Status")
    st.write(f"Estimated calls remaining: {st.session_state.api_calls_remaining}")
    minutes_since_reset = (datetime.now() - st.session_state.last_api_reset).total_seconds() / 60
    if minutes_since_reset > 60:  # Reset counter after an hour
        st.session_state.api_calls_remaining = 10
        st.session_state.last_api_reset = datetime.now()
    
    # Add manual refresh button in sidebar
    if st.button("Refresh Data Now"):
        st.session_state.refresh_counter += 1
        st.session_state.last_refresh = datetime.now()
        # Don't clear cache if running low on API calls
        if st.session_state.api_calls_remaining > 1:
            st.cache_data.clear()
    
    # Show refresh status
    st.info(f"Auto-refreshes every {refresh_interval} seconds")
    st.write(f"Last refreshed: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    st.write(f"Total refreshes: {st.session_state.refresh_counter}")

# Function to save data to local cache file
def save_to_cache(data):
    try:
        # Save to session state
        st.session_state.cached_data = data
        
        # Also save to file as backup
        cache_file = "crypto_cache.json"
        data_dict = data.to_dict(orient="records")
        with open(cache_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "data": data_dict
            }, f)
    except Exception as e:
        st.warning(f"Could not save cache: {e}")

# Function to load data from local cache
def load_from_cache():
    # First try session state
    if st.session_state.cached_data is not None:
        return st.session_state.cached_data
    
    # Then try file
    try:
        cache_file = "crypto_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                df = pd.DataFrame(cache_data["data"])
                cache_time = datetime.fromisoformat(cache_data["timestamp"])
                age_minutes = (datetime.now() - cache_time).total_seconds() / 60
                
                if age_minutes < 60:  # Use cache if less than 60 minutes old
                    return df
    except Exception as e:
        st.warning(f"Could not load cache: {e}")
    
    return None

# Function to fetch real-time crypto data from CoinGecko API with rate limit awareness
@st.cache_data(ttl=refresh_interval)
def fetch_crypto_data(limit=10, use_cache=False):
    # Check if we should use cache due to rate limits
    if st.session_state.api_calls_remaining < 2 or use_cache:
        cached_data = load_from_cache()
        if cached_data is not None:
            st.warning("⚠️ Using cached data due to API rate limits or fallback mode")
            return cached_data
    
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': limit,
        'page': 1,
        'sparkline': 'false',
        'price_change_percentage': '1h,24h,7d'
    }
    
    try:
        # Show spinner while fetching data
        with st.spinner("Fetching live cryptocurrency data..."):
            response = requests.get(url, params=params)
            
            # Update our rate limit tracking
            st.session_state.api_calls_remaining -= 1
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                
                # Check if we have the required columns
                required_columns = ['name', 'symbol', 'current_price', 'price_change_percentage_24h']
                if not all(col in df.columns for col in required_columns):
                    st.error("API response missing required columns")
                    cached_data = load_from_cache()
                    if cached_data is not None:
                        return cached_data
                    return pd.DataFrame()
                
                # Select and rename columns
                try:
                    df = df[['name', 'symbol', 'current_price', 'price_change_percentage_24h', 
                            'price_change_percentage_1h_in_currency', 
                            'price_change_percentage_24h_in_currency',
                            'price_change_percentage_7d_in_currency',
                            'market_cap', 'total_volume', 'circulating_supply']]
                    
                    df.columns = ['Coin', 'Symbol', 'Price (USD)', '24h Change (%)', 
                                '1h Change (%)', '24h Change (%)_alt', '7d Change (%)',
                                'Market Cap (USD)', 'Volume (24h)', 'Circulating Supply']
                    
                    # Save successful data to cache
                    save_to_cache(df)
                    return df
                except KeyError as e:
                    st.error(f"Error processing data: Missing column {e}")
                    cached_data = load_from_cache()
                    if cached_data is not None:
                        return cached_data
                    return pd.DataFrame()
            elif response.status_code == 429:
                st.error("⛔ API rate limit exceeded")
                st.session_state.api_calls_remaining = 0
                cached_data = load_from_cache()
                if cached_data is not None:
                    return cached_data
                return pd.DataFrame()
            else:
                st.error(f"API Error: {response.status_code}")
                cached_data = load_from_cache()
                if cached_data is not None:
                    return cached_data
                return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        cached_data = load_from_cache()
        if cached_data is not None:
            return cached_data
        return pd.DataFrame()

# Function to create and display dashboard content
def display_dashboard():
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    # Fetch the data
    df = fetch_crypto_data(limit=num_cryptos, use_cache=use_local_cache)
    
    if not df.empty:
        # Format percentage columns for display and color coding
        def color_negative_red(val):
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'
        
        # Add market trends and summary stats
        market_trend = "Bullish 📈" if df['24h Change (%)'].mean() > 0 else "Bearish 📉"
        highest_gainer = df.loc[df['24h Change (%)'].idxmax()]
        biggest_loser = df.loc[df['24h Change (%)'].idxmin()]
        
        # Create metrics row
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Market Trend", market_trend)
        with metric_cols[1]:
            st.metric("Top Performer (24h)", 
                      f"{highest_gainer['Coin']} ({highest_gainer['Symbol'].upper()})", 
                      f"{highest_gainer['24h Change (%)']:.2f}%")
        with metric_cols[2]:
            st.metric("Worst Performer (24h)", 
                      f"{biggest_loser['Coin']} ({biggest_loser['Symbol'].upper()})", 
                      f"{biggest_loser['24h Change (%)']:.2f}%")
        with metric_cols[3]:
            avg_change = df['24h Change (%)'].mean()
            st.metric("Average 24h Change", f"{avg_change:.2f}%")
        
        # Display the updated data in a table
        with col1:
            st.subheader("Current Prices")
            st.dataframe(
                df[['Coin', 'Symbol', 'Price (USD)', '24h Change (%)', 'Market Cap (USD)']]
                .style
                .format({
                    'Price (USD)': '${:,.2f}',
                    '24h Change (%)': '{:+.2f}%',
                    'Market Cap (USD)': '${:,.0f}'
                })
                .applymap(lambda x: color_negative_red(x) if isinstance(x, (int, float)) else '', 
                         subset=['24h Change (%)'])
                .bar(subset=['Market Cap (USD)'], color='#5fba7d'),
                use_container_width=True
            )
            
            # Time-based percentage changes
            st.subheader("Performance Overview")
            time_df = df[['Coin', '1h Change (%)', '24h Change (%)', '7d Change (%)']].copy()
            
            # Format and display the time-based changes
            st.dataframe(
                time_df.style.format({
                    '1h Change (%)': '{:+.2f}%',
                    '24h Change (%)': '{:+.2f}%',
                    '7d Change (%)': '{:+.2f}%'
                })
                .applymap(lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green' if isinstance(x, (int, float)) and x > 0 else '',
                         subset=['1h Change (%)', '24h Change (%)', '7d Change (%)']),
                use_container_width=True
            )

        # Plot the data
        with col2:
            # Price chart
            st.subheader("Price Comparison")
            if chart_type == "Bar":
                fig = px.bar(df, x='Coin', y='Price (USD)',
                             labels={'Coin': 'Cryptocurrency', 'Price (USD)': 'Price (USD)'},
                             color='Coin')
            else:  # Line chart
                fig = px.line(df, x='Coin', y='Price (USD)',
                              labels={'Coin': 'Cryptocurrency', 'Price (USD)': 'Price (USD)'},
                              markers=True)
            
            fig.update_layout(xaxis_title="Cryptocurrency", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)
            
            # 24h Change chart
            st.subheader("24h Price Change")
            fig_change = px.bar(df, x='Coin', y='24h Change (%)',
                               labels={'Coin': 'Cryptocurrency', '24h Change (%)': 'Change (%)'},
                               color='24h Change (%)',
                               color_continuous_scale=["red", "white", "green"],
                               range_color=[-max(abs(df['24h Change (%)'].min()), 
                                               abs(df['24h Change (%)'].max())), 
                                           max(abs(df['24h Change (%)'].min()), 
                                              abs(df['24h Change (%)'].max()))])
            
            fig_change.update_layout(xaxis_title="Cryptocurrency", yaxis_title="24h Change (%)")
            st.plotly_chart(fig_change, use_container_width=True)
        
        # Market dominance pie chart
        st.subheader("Market Cap Distribution")
        fig_pie = px.pie(df, values='Market Cap (USD)', names='Coin', 
                         title="Market Dominance")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Volume vs Market Cap scatter
        st.subheader("Volume vs. Market Cap")
        fig_scatter = px.scatter(df, x='Market Cap (USD)', y='Volume (24h)', 
                                size='Price (USD)', color='Coin',
                                hover_name='Coin', log_x=True, log_y=True,
                                labels={'Market Cap (USD)': 'Market Cap (log scale)', 
                                        'Volume (24h)': 'Trading Volume (log scale)'})
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Add last updated timestamp and data source note
        data_source = "Live CoinGecko API" if not use_local_cache and st.session_state.api_calls_remaining > 0 else "Cached Data"
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Source: {data_source}")
    else:
        st.error("No data available. Please check your internet connection or API limits.")

# Initial data load
display_dashboard()

# Create a placeholder for the auto-refresh mechanism
placeholder = st.empty()

# Real-time auto-refresh using Streamlit's native functionality
def auto_refresh():
    time_remaining = refresh_interval
    progress_bar = st.progress(0)
    refresh_status_text = st.empty()
    
    while time_remaining > 0:
        # Update progress bar and status text
        progress = 1 - (time_remaining / refresh_interval)
        progress_bar.progress(progress)
        refresh_status_text.text(f"Next refresh in {time_remaining} seconds...")
        
        # Sleep for 1 second
        time.sleep(1)
        time_remaining -= 1
    
    # Clear the progress elements
    progress_bar.empty()
    refresh_status_text.empty()
    
    # Increment refresh counter and update timestamp
    st.session_state.refresh_counter += 1
    st.session_state.last_refresh = datetime.now()
    
    # Only clear cache if not rate-limited
    if st.session_state.api_calls_remaining > 1 and not use_local_cache:
        st.cache_data.clear()
    
    # Trigger rerun
    st.rerun()

# Run the auto-refresh function if auto-refresh is enabled
with placeholder:
    auto_refresh()