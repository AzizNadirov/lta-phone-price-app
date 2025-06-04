"""
Phone Price Predictor Streamlit Application
------------------------------------------

This application allows users to:
1. Enter a URL of a phone product page
2. Extract phone specifications through web crawling
3. Predict the price based on these specifications
4. Compare the predicted price with the actual price from the source
5. Visualize price distributions for the phone model

Usage:
    streamlit run src/phoner.py
"""
import streamlit as st
import pandas as pd
import altair as alt
from loguru import logger
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

# Import our custom modules
from src.handlers import PhonerHandler
from src.schemas import PhoneSchema

# Configure logger
logger.add("logs/phoner_app.log", rotation="100 MB", retention="30 days")

# App configuration
st.set_page_config(
    page_title="Phone Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the handler (cached to avoid reloading on each rerun)
@st.cache_resource
def get_handler() -> PhonerHandler:
    """Initialize and cache the PhonerHandler instance"""
    logger.info("Initializing PhonerHandler")
    return PhonerHandler()

def display_phone_specs(phone_data: Dict[str, Any]) -> None:
    """
    Display the phone specifications in a structured format
    
    Args:
        phone_data: Dictionary containing phone specifications
    """
    # Check if we have an image link
    phone_image_link = phone_data.get('phone_image_link')
    
    # Create columns for layout (adjust column widths based on image presence)
    if phone_image_link:
        img_col, col1, col2 = st.columns([1, 1, 1])
    else:
        col1, col2 = st.columns(2)
    
    # Display phone image if available
    if phone_image_link:
        with img_col:
            st.image(phone_image_link, caption=f"{phone_data.get('brand', '')} {phone_data.get('model', '')}", 
                     use_container_width=True)
            
            # Add "View Full Size" link below the image
            st.markdown(f"[View Full Size Image]({phone_image_link})")
    
    with col1:
        st.subheader("Basic Information")
        st.markdown(f"**Brand:** {phone_data.get('brand', 'N/A')}")
        st.markdown(f"**Model:** {phone_data.get('model', 'N/A')}")
        st.markdown(f"**OS:** {phone_data.get('OS', 'N/A')}")
        st.markdown(f"**Source Price:** ${phone_data.get('price', 'N/A')}")
    
    with col2:
        st.subheader("Key Specifications")
        st.markdown(f"**RAM:** {phone_data.get('RAM_GB', 'N/A')} GB")
        st.markdown(f"**Storage:** {phone_data.get('ROM_GB', 'N/A')} GB")
        st.markdown(f"**Camera:** {phone_data.get('camera_mp', 'N/A')} MP")
        st.markdown(f"**CPU:** {phone_data.get('CPU_manufacturer', 'N/A')}")
        st.markdown(f"**NFC:** {'Yes' if phone_data.get('NFC') else 'No'}")
        
        # Display RAM/ROM ratio if available
        if 'RAM_ROM_ratio' in phone_data:
            st.markdown(f"**RAM/ROM Ratio:** {phone_data.get('RAM_ROM_ratio', 'N/A')}")
    
    # Show other specifications if available
    other_specs = phone_data.get('other_specifications', {})
    if other_specs and isinstance(other_specs, dict):
        with st.expander("Additional Specifications"):
            # Create a multi-column layout for other specs
            spec_cols = st.columns(2)
            
            # Distribute specs across columns
            for i, (key, value) in enumerate(sorted(other_specs.items())):
                col_idx = i % 2
                with spec_cols[col_idx]:
                    # Format the key for better readability
                    formatted_key = key.replace('_', ' ').title()
                    st.markdown(f"**{formatted_key}:** {value}")
    
    # Show all specifications in a table format
    with st.expander("All Specifications (Table View)"):
        # Filter out any None values for cleaner display
        clean_data = {k: v for k, v in phone_data.items() if v is not None and k != 'other_specifications'}
        
        # Add flattened other_specifications if available
        if other_specs and isinstance(other_specs, dict):
            for key, value in other_specs.items():
                formatted_key = f"spec_{key}"
                clean_data[formatted_key] = value
                
        specs_df = pd.DataFrame([clean_data])
        st.dataframe(specs_df)

def display_price_comparison(actual_price: float, predicted_price: float) -> None:
    """
    Display a comparison between actual and predicted prices
    
    Args:
        actual_price: The actual price from the source
        predicted_price: The predicted price from the ML model
    """
    st.subheader("Price Comparison")
    
    # Calculate difference and percentage
    diff = predicted_price - actual_price
    diff_percentage = (diff / actual_price) * 100 if actual_price else 0
    
    # Create metrics for price comparison
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Source Price", f"${actual_price:.2f}")
    with col2:
        st.metric("Predicted Price", f"${predicted_price:.2f}")
    with col3:
        st.metric(
            "Difference", 
            f"${abs(diff):.2f}", 
            f"{diff_percentage:.1f}%" if diff > 0 else f"{diff_percentage:.1f}%",
            delta_color="inverse"
        )
    
    # Add interpretation
    if abs(diff_percentage) <= 10:
        st.success("✅ The prediction is very close to the actual price (within 10%).")
        st.markdown("""
        **This indicates:**
        - The phone is fairly priced according to market standards
        - Specifications align well with typical market value
        - This model follows expected pricing patterns
        """)
    elif abs(diff_percentage) <= 20:
        st.info("ℹ️ The prediction has moderate accuracy (within 20%).")
        st.markdown("""
        **This suggests:**
        - The phone may have some unique features affecting its price
        - The model or brand may command a slight premium or discount
        - Consider the specific features that might explain this difference
        """)
    else:
        st.warning("⚠️ There's a significant difference between predicted and actual prices.")
        if diff > 0:
            st.markdown("""
            **This phone may be overpriced because:**
            - Brand premium exceeds the value of technical specifications
            - It might include features our model doesn't account for
            - There could be limited supply or high demand affecting market price
            - Consider comparing with other models with similar specifications
            """)
        else:
            st.markdown("""
            **This phone may be underpriced because:**
            - It could be on sale or promotional pricing
            - Older model being cleared from inventory
            - May lack features not captured in basic specifications
            - Consider checking if this represents good value for money
            """)
            
    # Add a chart to visually represent the comparison
    chart_data = pd.DataFrame({
        'Type': ['Source Price', 'Predicted Price'],
        'Price': [actual_price, predicted_price]
    })
    
    price_chart = alt.Chart(chart_data).mark_bar().encode(
        x='Type',
        y='Price',
        color=alt.condition(
            alt.datum.Type == 'Source Price',
            alt.value('steelblue'),
            alt.value('orange')
        )
    ).properties(
        height=200
    )
    
    st.altair_chart(price_chart, use_container_width=True)

def display_price_distribution(handler: PhonerHandler, model: str) -> None:
    """
    Display the price distribution for the given phone model
    
    Args:
        handler: The PhonerHandler instance
        model: The phone model name
    """
    st.subheader("Price Distribution Analysis")
    
    # Get the price distribution chart
    fig = handler.ml.plot_stats_for_model(model=model)
    
    if fig:
        st.pyplot(fig)
        
        # Add explanations for the visualization methods
        with st.expander("How to Interpret These Visualizations"):
            st.markdown("""
            ### Understanding the Price Distribution Charts
            
            #### 1. Histogram with KDE Overlay
            Shows the frequency distribution of prices for this model. The curve (Kernel Density Estimate) represents the overall shape of the distribution - peaks indicate common price points.
            
            #### 2. Box Plot
            Displays the price distribution using:
            - The box: Middle 50% of prices (25th to 75th percentile)
            - The line in the box: Median price
            - Whiskers: Price range (excluding outliers)
            - Points outside whiskers: Outlier prices
            
            #### 3. Violin Plot
            Similar to a box plot but shows the full probability distribution. Wider sections indicate more phones are available at that price point.
            
            #### 4. Statistical Summary Table
            Provides numerical statistics for each model:
            - Mean: Average price
            - Median: Middle price point (50% of phones cost more, 50% cost less)
            - Std Dev: How spread out the prices are (higher means more variability)
            - Min/Max: Lowest and highest prices observed
            - Count: Number of data points for this model
            
            #### 5. Q-Q Plot
            Assesses if price distribution follows a normal bell curve. Points following the diagonal line suggest normal distribution.
            
            #### 6. Cumulative Distribution Function (CDF)
            Shows the probability that a phone costs less than or equal to a given price. Useful for determining price percentiles.
            
            ### What This Means For Your Phone
            - If the current phone's price is far from the mean/median, it may be over/underpriced
            - High standard deviation suggests inconsistent pricing for this model
            - Compare the predicted price position on these distributions to evaluate market positioning
            """)
        
        st.caption("This analysis shows the distribution of prices for similar models in our database.")
    else:
        st.info("No historical price data available for this model in our database.")

def save_to_history(phone_data: Dict[str, Any], predicted_price: float) -> None:
    """
    Save the analyzed phone to session history
    
    Args:
        phone_data: The phone specifications
        predicted_price: The predicted price
    """
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Create a history entry with key information
    history_entry = {
        "brand": phone_data.get("brand", "Unknown"),
        "model": phone_data.get("model", "Unknown"),
        "actual_price": phone_data.get("price", 0),
        "predicted_price": predicted_price,
        "url": st.session_state.get("last_url", "")
    }
    
    # Add to history (avoid duplicates)
    if not any(entry["model"] == history_entry["model"] and 
               entry["actual_price"] == history_entry["actual_price"] 
               for entry in st.session_state.history):
        st.session_state.history.append(history_entry)

def show_history() -> None:
    """Display the history of analyzed phones"""
    if "history" in st.session_state and st.session_state.history:
        st.subheader("Analysis History")
        
        # Create a DataFrame from history
        history_df = pd.DataFrame(st.session_state.history)
        
        # Add a difference column
        history_df["diff_percentage"] = ((history_df["predicted_price"] - history_df["actual_price"]) / 
                                        history_df["actual_price"] * 100).round(2)
        
        # Format for display
        display_df = history_df.copy()
        display_df["actual_price"] = display_df["actual_price"].apply(lambda x: f"${x:.2f}")
        display_df["predicted_price"] = display_df["predicted_price"].apply(lambda x: f"${x:.2f}")
        display_df["diff_percentage"] = display_df["diff_percentage"].apply(lambda x: f"{x}%")
        
        st.dataframe(display_df)
        
        # Add a clear history button
        if st.button("Clear History", key="clear_history_button", type="secondary"):
            st.session_state.history = []
            st.experimental_rerun()

def main() -> None:
    """Main function to run the Streamlit application"""
    
    # App title and description
    st.title("📱 Phone Price Predictor")
    st.markdown("""
    This application crawls phone specifications from a provided URL,
    predicts the price based on machine learning models, and compares
    it with the actual price found on the page.
    """)
    
    # Initialize handler
    handler = get_handler()
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        ### How it works
        
        This app uses machine learning to predict phone prices based on specifications:
        
        1. **Data Collection**: The model is trained on a dataset of phone specifications and their prices
        2. **Feature Engineering**: Important features like RAM, storage, camera quality, etc. are extracted
        3. **Prediction**: A regression model predicts the price based on the phone's specifications
        4. **Comparison**: The predicted price is compared with the actual price from the source
        """)
        
        # Add some spacing
        st.markdown("---")
        
        # Show the history in the sidebar
        show_history()
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        ### How it works
        
        This app uses machine learning to predict phone prices based on specifications:
        
        1. **Data Collection**: The model is trained on a dataset of phone specifications and their prices
        2. **Feature Engineering**: Important features like RAM, storage, camera quality, etc. are extracted
        3. **Prediction**: A regression model predicts the price based on the phone's specifications
        4. **Comparison**: The predicted price is compared with the actual price from the source
        
        The model is periodically retrained with new data to keep predictions accurate.
        """)
        
        # Allow manual model retraining
        if st.button("Retrain ML Model", help="Manually trigger retraining of the ML model"):
            with st.spinner("Retraining model... This may take a while"):
                try:
                    handler.ml.train()
                    st.success("✅ Model retrained successfully!")
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    logger.exception("Model retraining failed")
        
        # Add some spacing
        st.markdown("---")
        
        # Show the history in the sidebar
        show_history()
    
    # URL input section
    st.header("Enter Phone Product URL")
    url = st.text_input(
        "URL:", 
        placeholder="https://example.com/phone/product",
        help="Enter the URL of a phone product page to analyze"
    )
    
    # Save URL to session state for history
    if url:
        st.session_state.last_url = url
    
    # Process button
    if st.button("Analyze Phone", type="primary"):
        if not url:
            st.error("⚠️ Please enter a valid URL")
        else:
            # Show processing status
            with st.spinner("Crawling and analyzing the phone data..."):
                try:
                    # Log the action
                    logger.info(f"Processing URL: {url}")
                    
                    # Use the handler's method to get prediction from URL
                    predicted_price = handler.predict_from_url(url=url)
                    
                    # Get the parsed data by calling the crawler again since we don't have direct access to the last parsed data
                    parsed_data = handler.crawler.start_url(url=url)
                    
                    if not parsed_data or predicted_price is None:
                        st.error("❌ Failed to extract phone specifications or predict price from the provided URL.")
                        st.markdown("""
                        Possible reasons:
                        - The URL doesn't contain a phone product page
                        - The website structure isn't compatible with our crawler
                        - The extracted data doesn't match our schema requirements
                        
                        Try a different URL or contact support if the issue persists.
                        """)
                        return
                    
                    # Display success message
                    st.success("✅ Successfully extracted phone data and predicted price!")
                    
                    # Create a container for results
                    with st.container():
                        # Display the parsed specifications
                        st.header("Phone Analysis Results")
                        display_phone_specs(parsed_data)
                        
                        # Get the actual price from parsed data
                        actual_price = parsed_data.get("price", 0)
                        
                        # Display price comparison
                        display_price_comparison(actual_price, predicted_price)
                        
                        # Display price distribution
                        model_name = parsed_data.get("model", "")
                        if model_name:
                            display_price_distribution(handler, model_name)
                        
                        # Save to history
                        save_to_history(parsed_data, predicted_price)
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    logger.exception(f"Error processing URL: {url}")
                    
                    # Provide more detailed error information in an expander
                    with st.expander("Error Details"):
                        st.code(str(e))
                        st.markdown("""
                        If you're seeing this error repeatedly, please try:
                        1. Checking if the URL is correctly formatted and accessible
                        2. Verifying that the page contains phone specifications
                        3. Trying a different phone product page
                        """)
    
    # Example URLs section
    with st.expander("Example URLs to Try"):
        st.markdown("""
        Here are some example URLs you can try:
        
        - `https://www.bakuelectronics.az/catalog/telefonlar-qadcetler/smartfonlar-mobil-telefonlar/xiaomi-poco-m5-6gb128gb-black.html`
        - `https://kontakt.az/honor-400-lite-8-256-gb-grey`
        - `https://mgstore.az/oppo-a3x-4-128-gb-ocean-blue`
        - `https://www.amazon.com/dp/B0CHX1W1XY` (Amazon - iPhone 15)
        - `https://www.bestbuy.com/site/samsung-galaxy-s23-128gb-unlocked-phantom-black/6529758.p` (Best Buy)
        
        Note: The crawler works best with product pages that have clear specifications sections.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2055 CoolGuys | Developed with Sonnet 3.7:D")

if __name__ == "__main__":
    main()