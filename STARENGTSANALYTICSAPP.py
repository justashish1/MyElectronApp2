import streamlit as st
import pandas as pd
import base64
import numpy as np
import time
import io
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import logging
from st_aggrid import AgGrid, GridOptionsBuilder

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logging.info('Application started')

# Set the page configuration
st.set_page_config(page_title="Starengts Timeseries Analysis Application", layout="wide")

# Generate time options
def generate_time_options():
    return [f"{hour:02d}:{minute:02d}:00" for hour in range(24) for minute in range(60)]

# Load logo as base64
def load_logo(filename):
    with open(filename, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_image}"

# Custom CSS for styling
def custom_css():
    st.markdown("""
        <style>
            .main-title {
                font-size: 25px;
                color: #32c800;
                text-align: center;
                font-weight: bold;
            }
            .current-time {
                font-size: 18px;
                font-weight: bold;
                display: inline-block;
                margin-right: 20px;
            }
            .upload-button {
                width: 30%;
                height: 50px;
                line-height: 50px;
                border-width: 1px;
                border-style: dashed;
                border-radius: 5px;
                text-align: center;
                margin-bottom: 5px;
                font-size: 10px;
                position: relative;
                top: 5px;
                left: 10px;
                color: #32c800;
            }
            .center-text {
                text-align: center;
            }
            .logo {
                height: 35px;
                display: inline-block;
                margin-left: auto;
                margin-right: 10px;
            }
            .header {
                position: relative;
                width: 100%;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                color: #32c800;
                align-items: center;
            }
            .stProgress > div > div > div > div {
                background-color: #32c800;
            }
            .stButton > button {
                background-color: #32c800;
                border: none;
            }
            .stButton > button:hover {
                background-color: #28a745;
            }
            .custom-error {
                background-color: #ff4c4c;
                color: white;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

# Get the current time as a string
def get_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Display the logo and time
def display_logo_and_time(logo_src):
    current_time_html = f"""
        <div class='header'>
            <div class='current-time' id='current-time'>{get_time()}</div>
            <img src='{logo_src}' class='logo'>
        </div>
    """
    st.markdown(current_time_html, unsafe_allow_html=True)

# Add JavaScript for live clock
def add_js_script():
    st.markdown("""
        <script>
        document.addEventListener('DOMContentLoaded', (event) => {
            function updateTime() {
                var now = new Date();
                var timeString = now.getFullYear() + '-' + 
                                 ('0' + (now.getMonth()+1)).slice(-2) + '-' + 
                                 ('0' + now.getDate()).slice(-2) + ' ' + 
                                 ('0' + now.getHours()).slice(-2) + ':' + 
                                 ('0' + now.getMinutes()).slice(-2) + ':' + 
                                 ('0' + now.getSeconds()).slice(-2);
                document.getElementById('current-time').innerHTML = timeString;
            }
            setInterval(updateTime, 1000);
        });
        </script>
    """, unsafe_allow_html=True)

# Authenticate user
def authenticate(username, password):
    if username == "admin" and password == "password106":
        st.session_state.authenticated = True
    else:
        st.error("Invalid username or password")

# Load data from uploaded file
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)

# Preprocess data
@st.cache_data
def preprocess_data(df):
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    return df

# Validate 'Timestamp' column and format
def validate_timestamp_column(df):
    if 'Timestamp' not in df.columns:
        st.markdown('<div class="custom-error">The uploaded file does not contain a \'Timestamp\' column The correct column name is Timestamp and format is YYYY-MM-DD HH:MM:SS.</div>', unsafe_allow_html=True)
        logging.error("The uploaded file does not contain a 'Timestamp' column, The correct column name is Timestamp and format is YYYY-MM-DD HH:MM:SS")
        st.write("### Debugging Information")
        st.write(df.head())  # Display the first few rows of the dataframe for debugging
        return False
    try:
        pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        st.markdown('<div class="custom-error">The \'Timestamp\' column is not in the correct datetime format (YYYY-MM-DD HH:MM:SS).</div>', unsafe_allow_html=True)
        logging.error("The 'Timestamp' column is not in the correct datetime format (YYYY-MM-DD HH:MM:SS).")
        st.write("### Debugging Information")
        st.write(df.head())  # Display the first few rows of the dataframe for debugging
        return False
    return True

# Resample dataframe
@st.cache_data
def get_resampled_df(filtered_df, sampling_interval):
    return filtered_df.resample(f'{sampling_interval}min').mean().fillna(0)

# Generate forecast using Prophet
def generate_forecast(df, value_column, periods):
    df.reset_index(inplace=True)
    df.rename(columns={'Timestamp': 'ds', value_column: 'y'}, inplace=True)
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Display data using st-aggrid
def display_aggrid(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True) # Add pagination
    gb.configure_side_bar() # Add a sidebar
    gridOptions = gb.build()
    AgGrid(
        df,
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        height=400,
        width='100%',
        fit_columns_on_grid_load=True
    )

# Main function
def main():
    custom_css()
    logo_src = load_logo('logo.png')
    display_logo_and_time(logo_src)
    add_js_script()
    st.markdown("<h1 class='main-title'>STARENGTS TIMESERIES ANALYSIS APPLICATION</h1>", unsafe_allow_html=True)

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            authenticate(username, password)
            if st.session_state.authenticated:
                st.experimental_rerun()
        st.stop()

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"], label_visibility="visible", help="Upload a file in CSV or Excel format")

    if uploaded_file:
        # Removing redundant progress bar
        st.markdown("""
            <style>
                .stProgress > div > div > div > div {
                    background-color: #32c800;
                }
            </style>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        start_time = datetime.now()
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        df = load_data(uploaded_file)
        process_end_time = datetime.now()
        loading_time = (process_end_time - start_time).total_seconds()
        st.write(f"Loaded file: {uploaded_file.name} (Total processing time: {loading_time:.2f} seconds)")
        logging.info(f"Loaded file: {uploaded_file.name} (Total processing time: {loading_time:.2f} seconds)")

        if not validate_timestamp_column(df):
            return

        df = preprocess_data(df)

        if not df.empty:
            # Filter section
            st.markdown('<div>', unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                start_date = st.date_input("Start Date", min_value=df.index.min().date(), max_value=df.index.max().date(), value=df.index.min().date())

            with col2:
                end_date = st.date_input("End Date", min_value=df.index.min().date(), max_value=df.index.max().date(), value=df.index.max().date())

            with col3:
                time_options = generate_time_options()
                start_time_index = time_options.index(df.index.min().strftime('%H:%M:%S'))
                start_time_str = st.selectbox("Start Time", time_options, index=start_time_index)

            with col4:
                end_time_index = time_options.index(df.index.max().strftime('%H:%M:%S'))
                end_time_str = st.selectbox("End Time", time_options, index=end_time_index)

            with col5:
                value_column = st.selectbox("Value Column", [col for col in df.columns if col != 'Timestamp'])

            with col6:
                sampling_interval = st.slider("Sampling Interval (minutes)", 1, 60, 1)
            st.markdown('</div>', unsafe_allow_html=True)

            # Rest of the content
            st.markdown('<div class="content">', unsafe_allow_html=True)
            
            for col in df.select_dtypes(include=['category', 'object']).columns:
                unique_values = df[col].unique()
                selected_values = st.multiselect(f"Filter by {col}", unique_values, default=unique_values)
                df = df[df[col].isin(selected_values)]

            start_datetime = datetime.strptime(f"{start_date} {start_time_str}", '%Y-%m-%d %H:%M:%S')
            end_datetime = datetime.strptime(f"{end_date} {end_time_str}", '%Y-%m-%d %H:%M:%S')
            mask = (df.index >= start_datetime) & (df.index <= end_datetime)
            filtered_df = df.loc[mask]
            resampled_df = get_resampled_df(filtered_df, sampling_interval)

            # Add font size option
            font_size = st.selectbox("Select text size", [10, 12, 14, 16, 18], index=1)

            # Apply selected font size
            st.markdown(
                f"""
                <style>
                .stText, .stDataFrame {{
                    font-size: {font_size}px;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("## DataFrame Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.write("### Head")
                st.write(resampled_df.head())

            with col2:
                st.write("### Info")
                buffer = io.StringIO()
                resampled_df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

            with col3:
                st.write("### Shape")
                st.write(resampled_df.shape)

            with col4:
                st.write("### Size")
                st.write(resampled_df.size)

            # Anomaly detection
            isolation_forest = IsolationForest(contamination=0.05)
            anomalies = isolation_forest.fit_predict(resampled_df[[value_column]])
            resampled_df['Anomaly'] = anomalies

            # Plot the time series data
            fig = go.Figure()

            inactivity_mask = (resampled_df[value_column].rolling('10min').max() - resampled_df[value_column].rolling('10min').min()) <= 15
            active_df = resampled_df[~inactivity_mask]
            inactive_df = resampled_df[inactivity_mask]

            fig.add_trace(go.Scatter(x=active_df.index, y=active_df[value_column], mode='lines', line=dict(color='blue'), name='Active Periods', connectgaps=True))
            fig.add_trace(go.Scatter(x=inactive_df.index, y=inactive_df[value_column], mode='lines', line=dict(color='red'), name='Inactivity Periods', connectgaps=True))
            fig.add_trace(go.Scatter(x=resampled_df[resampled_df['Anomaly'] == -1].index, y=resampled_df[resampled_df['Anomaly'] == -1][value_column], mode='markers', name='Anomalies', marker=dict(color='orange')))

            X = np.array((resampled_df.index - resampled_df.index.min()).total_seconds()).reshape(-1, 1)
            y = resampled_df[value_column].values
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)

            fig.add_trace(go.Scatter(x=resampled_df.index, y=y_pred, mode='lines', line=dict(color='green', dash='dash'), name='Regression Line'))

            fig.update_layout(title='Time Series Data with Inactivity Periods, Anomalies, and Regression Line', xaxis_title='Timestamp', yaxis_title=value_column)
            st.plotly_chart(fig)
            st.markdown("**The time series plot displays the data over time, with blue lines representing active periods, red lines indicating inactivity periods, and orange markers highlighting anomalies. The green dashed line shows the linear regression line, which helps identify the overall trend in the data.**")

            box_fig = go.Figure()
            box_fig.add_trace(go.Box(y=resampled_df[value_column], name=value_column, boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color='blue'))
            st.plotly_chart(box_fig)
            st.markdown("**The box plot visualizes the distribution of the selected data. It displays the median (line inside the box), the interquartile range (the box), and potential outliers (points outside the whiskers). The box plot helps identify the central tendency and variability of the data.**")

            decomposition = seasonal_decompose(resampled_df[value_column], model='additive', period=30)
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal.dropna()
            resid = decomposition.resid.dropna()

            decomposition_fig = go.Figure()
            decomposition_fig.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend', line=dict(color='blue')))
            decomposition_fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonality', line=dict(color='orange')))
            decomposition_fig.add_trace(go.Scatter(x=resid.index, y=resid, mode='lines', name='Residuals', line=dict(color='green')))
            st.plotly_chart(decomposition_fig, use_container_width=True)
            st.markdown("**The time series decomposition plot breaks down the data into its trend, seasonal, and residual components. The trend component shows the long-term direction, the seasonal component captures repeating patterns, and the residual component represents random noise.**")

            control_chart_fig = go.Figure()
            control_chart_fig.add_trace(go.Scatter(x=resampled_df.index, y=resampled_df[value_column], mode='lines', name='Load cell Value', line=dict(color='blue')))
            control_chart_fig.add_trace(go.Scatter(x=resampled_df.index, y=resampled_df[value_column].rolling(window=30).std(), mode='lines', name='Rolling Std', line=dict(color='orange'), yaxis='y2'))
            control_chart_fig.update_layout(title='Control Charts (X-bar and R charts)', xaxis_title='Timestamp', yaxis=dict(title=value_column), yaxis2=dict(title='Standard Deviation', overlaying='y', side='right'))
            st.plotly_chart(control_chart_fig, use_container_width=True)
            st.markdown("**The control chart monitors the process stability over time. The X-bar chart shows the mean of the process, and the R chart displays the range of the process variation. These charts help identify any unusual variations in the process.**")

            kmeans = KMeans(n_clusters=3)
            resampled_df['Cluster'] = kmeans.fit_predict(resampled_df[[value_column]])
            num_clusters = len(set(resampled_df['Cluster']))

            if num_clusters > 1:
                silhouette_avg = silhouette_score(resampled_df[[value_column]], resampled_df['Cluster'])
            else:
                silhouette_avg = 'N/A'

            cluster_fig = go.Figure()
            colors = ['blue', 'orange', 'green']
            for cluster in range(num_clusters):
                cluster_data = resampled_df[resampled_df['Cluster'] == cluster]
                cluster_fig.add_trace(go.Scatter(x=cluster_data.index, y=cluster_data[value_column], mode='markers', marker=dict(color=colors[cluster]), name=f'Cluster {cluster}'))

            cluster_fig.update_layout(title=f'KMeans Clustering (Silhouette Score: {silhouette_avg})', xaxis_title='Timestamp', yaxis_title=value_column)
            st.plotly_chart(cluster_fig, use_container_width=True)
            st.markdown("**The clustering plot uses KMeans to group the data into clusters. Each color represents a different cluster, helping to identify patterns and similarities within the data. The silhouette score indicates how well the data points fit within their clusters, with higher values representing better clustering.**")

            stats = resampled_df[value_column].describe(percentiles=[.25, .5, .75])

            total_active_time = active_df.shape[0] * sampling_interval
            total_inactive_time = inactive_df.shape[0] * sampling_interval

            r_squared = reg.score(X, y)

            stats_output = f"""
            **Descriptive Statistics**
            - Mean: {stats['mean']:.2f}
            - Max: {stats['max']:.2f}
            - Min: {stats['min']:.2f}
            - Standard Deviation: {stats['std']:.2f}
            - 25%: {stats['25%']:.2f}
            - 50% (Median): {stats['50%']:.2f}
            - 75%: {stats['75%']:.2f}
            **Linear Regression**
            - Equation: y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}
            - R²: {r_squared:.2f}
            **Activity Duration**
            - Total Active Time: {total_active_time} minutes
            - Total Inactive Time: {total_inactive_time} minutes
            """
            st.markdown(stats_output)
            logging.info("Descriptive statistics generated.")

            st.write("## Additional Visualizations")
            st.write("### Histogram")
            fig_hist = go.Figure()
            colors = px.colors.qualitative.Plotly

            for i, col in enumerate(df.columns):
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig_hist.add_trace(go.Histogram(x=df[col], name=col, marker=dict(color=colors[i % len(colors)]), opacity=0.75))

            fig_hist.update_layout(barmode='overlay', title='Histogram of Numeric Columns', xaxis_title='Value', yaxis_title='Count', legend=dict(x=1, y=1, traceorder='normal'), bargap=0.2)
            fig_hist.update_traces(opacity=0.75)
            st.plotly_chart(fig_hist, use_container_width=True)
            logging.info("Histogram generated.")
            st.markdown("**The histogram visualizes the distribution of the data for each numeric column in the dataset.**")

            st.write("### Pair Plot")
            pair_plot_fig = sns.pairplot(df.select_dtypes(include=[np.number]), diag_kind='kde')
            st.pyplot(pair_plot_fig)
            logging.info("Pair plot generated.")
            st.markdown("**The pair plot displays pairwise relationships in the dataset, showing scatter plots for each pair of features and histograms for individual features.**")

            st.write("### Correlation Heatmap")
            corr = df.corr()
            fig_heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index.values, y=corr.columns.values, colorscale='Viridis'))
            fig_heatmap.update_layout(title='Correlation Heatmap')
            st.plotly_chart(fig_heatmap, use_container_width=True)
            logging.info("Correlation heatmap generated.")
            st.markdown("**The correlation heatmap displays the correlation coefficients between pairs of features in the dataset. The colors represent the strength of the correlations.**")

            st.write("### User Annotations")
            annotation_text = st.text_input("Enter annotation text")
            annotation_x = st.text_input("Enter x value for annotation")
            annotation_y = st.text_input("Enter y value for annotation")
            if st.button("Add Annotation"):
                fig.add_trace(go.Scatter(x=[annotation_x], y=[annotation_y], mode='text', text=[annotation_text], name='Annotation'))
                st.plotly_chart(fig)

            st.write("### Advanced Analytics")
            degree = st.slider("Degree of Polynomial Regression", 1, 10, 2)
            poly_features = np.polyfit(X.flatten(), y, degree)
            poly_model = np.poly1d(poly_features)
            y_poly_pred = poly_model(X.flatten())
            fig.add_trace(go.Scatter(x=resampled_df.index, y=y_poly_pred[:len(resampled_df.index)], mode='lines', line=dict(color='purple', dash='dot'), name=f'Polynomial Regression (degree {degree})'))
            st.plotly_chart(fig)

            forecast_periods = st.number_input("Forecasting Period (days)", min_value=1, max_value=365, value=180)

            if st.button("Forecast Future", key="forecast_future"):
                with st.spinner('Forecasting...'):
                    try:
                        forecast = generate_forecast(resampled_df, value_column, forecast_periods)
                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))
                        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line=dict(color='gray'), showlegend=False))
                        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='gray'), showlegend=False))
                        st.plotly_chart(fig_forecast)
                        logging.info("Forecast plot generated.")
                        st.markdown("**The forecast plot shows the predicted future values of the selected time series. The blue line represents the forecasted values, while the shaded area indicates the uncertainty intervals (upper and lower bounds). The forecast helps in understanding the potential future trends based on the historical data.**")

                        csv = forecast.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">Download Forecast Data</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        logging.info("Forecast data download link generated.")
                    except Exception as e:
                        st.error(f"Forecasting failed: {e}")
                        logging.error(f"Forecasting failed: {e}")

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="custom-error">The uploaded file does not contain a \'Timestamp\' column.</div>', unsafe_allow_html=True)
            st.write("### Debugging Information")
            st.write(df.head())  # Display the first few rows of the dataframe for debugging
            logging.error("The uploaded file does not contain a 'Timestamp' column,The correct column name is Timestamp and format is YYYY-MM-DD HH:MM:SS.")
    else:
        st.write("Please upload a CSV or Excel file to get started.")
        logging.info("Waiting for file upload.")

if __name__ == "__main__":
    main()
