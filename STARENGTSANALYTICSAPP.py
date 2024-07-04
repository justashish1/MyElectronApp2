import streamlit as st
import pandas as pd
import base64
import numpy as np
import time
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
import io
import os
import pytz

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logging.info('Application started')

# Set the page configuration
st.set_page_config(page_title="STARENGTS Timeseries Analysis Application", layout="wide")

# Generate time options
def generate_time_options():
    return [f"{hour % 12 if hour % 12 else 12:02d}:{minute:02d}:{second:02d} {'AM' if hour < 12 else 'PM'}"
            for hour in range(24) for minute in range(60) for second in range(60)]

# Load logo as base64
def load_logo(filename):
    with open(filename, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_image}"

# Developer info at the bottom left
st.markdown("""
    <div class='developer-info'>
        Developer Name : Ashish Malviya<br>
    </div>
""", unsafe_allow_html=True)

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
            .current-date {
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
                height: 45px;
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
            .developer-info {
                position: fixed;
                bottom: 0;
                left: 0;
                text-align:left;
                margin: 10px;
                font-size: 12px;
            }
            .stProgress > div > div > div > div {
                background-color: #32c800;
            }
            .content {
                padding-top: 0px;
            }
            .stButton > button {
                background-color: #32c800;
                color: white;
                border: none;
                font-weight: bold;
            }
            .stButton > button:hover {
                color: white;
                background-color: #32c800;
            }
            .custom-error {
                background-color: #32c800;
                color: white;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            .df-overview-title {
                font-size: 16px;
                font-weight: bold;
            }
            .df-overview-section {
                font-size: 16px;
                font-weight: bold;
                color: black.
            }
            .df-shape-size {
            }
            .download-manual {
                font-size: 18px;
                font-weight: bold.
            }
            .additional-visualizations {
                font-size: 20px;
                font-weight: bold;
                margin-top: 20px.
            }
            .histogram, .user-annotations, .advanced-analytics, .correlation-heatmap, .pair-plot {
                font-size: 18px;
                font-weight: bold.
            }
            .outlier-treatment {
                font-size: 18px;
                font-weight: bold.
                margin-top: 20px;
                margin-bottom: 20px.
            }
            .spacing {
                margin-top: 50px.
            }
            .left-side, .right-side {
                height: 100%;
                overflow-y: auto;
            }
        </style>
    """, unsafe_allow_html=True)

# Function to get the current date as a string for the clock
def get_date(timezone_str='UTC'):
    tz = pytz.timezone(timezone_str)
    return datetime.now(tz).strftime('%Y-%m-%d')

# Display the logo and date
def display_logo_and_date(logo_src, timezone_str):
    current_date_html = f"""
        <div class='header'>
            <div class='current-date' id='current-date'>{get_date(timezone_str)}</div>
            <img src='{logo_src}' class='logo'>
        </div>
    """
    st.markdown(current_date_html, unsafe_allow_html=True)

# Add JavaScript for live date and timezone detection
def add_js_script():
    st.markdown("""
        <script>
        document.addEventListener('DOMContentLoaded', (event) => {
            function updateDate() {
                var now = new Date();
                var dateString = now.getFullYear() + '-' + 
                                 ('0' + (now.getMonth() + 1)).slice(-2) + '-' + 
                                 ('0' + now.getDate()).slice(-2);
                document.getElementById('current-date').innerHTML = dateString;
            }
            setInterval(updateDate, 1000);

            var timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
            var tzElement = document.createElement('input');
            tzElement.type = 'hidden';
            tzElement.id = 'timezone';
            tzElement.value = timezone;
            document.body.appendChild(tzElement);
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
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %I:%M:%S.%f %p', errors='coerce')
        df = df.dropna(subset=['DateTime'])
        df.set_index('DateTime', inplace=True)
    return df

# Validate 'DateTime' column and format
def validate_datetime_column(df):
    if 'DateTime' not in df.columns:
        st.markdown('<div class="custom-error">The uploaded file does not contain a \'DateTime\' column The correct column name is DateTime and format is DD/MM/YYYY hh:mm:ss.SSS AM/PM.</div>', unsafe_allow_html=True)
        logging.error("The uploaded file does not contain a 'DateTime' column, The correct column name is DateTime and format is DD/MM/YYYY hh:mm:ss.SSS AM/PM")
        st.write("### Debugging Information")
        st.write(df.head())  # Display the first few rows of the dataframe for debugging
        return False
    try:
        pd.to_datetime(df['DateTime'], format='%d/%m/%Y %I:%M:%S.%f %p')
    except ValueError:
        st.markdown('<div class="custom-error">The \'DateTime\' column is not in the correct datetime format (DD/MM/YYYY hh:mm:ss.SSS AM/PM).</div>', unsafe_allow_html=True)
        logging.error("The 'DateTime' column is not in the correct datetime format (DD/MM/YYYY hh:mm:ss.SSS AM/PM).")
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
    df.rename(columns={'DateTime': 'ds', value_column: 'y'}, inplace=True)
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Display data using streamlit-aggrid
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

# Function to download the manual
def download_manual():
    manual_path = "Applications_manual.pdf"  # File in the same directory
    if os.path.exists(manual_path):
        with open(manual_path, "rb") as file:
            manual_data = file.read()
        file_name = os.path.basename(manual_path)
        b64 = base64.b64encode(manual_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{file_name}"> {file_name}</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Manual not available. Send request to Ashish Malviya!")

# Outlier treatment function
def treat_outliers(df, value_column):
    Q1 = df[value_column].quantile(0.25)
    Q3 = df[value_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    treated_df = df[(df[value_column] >= lower_bound) & (df[value_column] <= upper_bound)]
    return treated_df

# Main function
def main():
    custom_css()
    logo_src = load_logo('logo.png')
    add_js_script()

    # Get the timezone from the hidden HTML element using Streamlit's query_params
    timezone = st.query_params.get('timezone', ['UTC'])[0]

    display_logo_and_date(logo_src, timezone)
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

    with st.sidebar:
        st.markdown("<h2>Upload Data</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"], label_visibility="visible", help="Upload a file in CSV or Excel format")
        if uploaded_file:
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

            if not validate_datetime_column(df):
                return

            df = preprocess_data(df)

            st.session_state.df = df

            start_date = st.date_input("Start Date", min_value=df.index.min().date(), max_value=df.index.max().date(), value=df.index.min().date())
            end_date = st.date_input("End Date", min_value=df.index.min().date(), max_value=df.index.max().date(), value=df.index.max().date())
            time_options = generate_time_options()
            start_time_index = time_options.index(df.index.min().strftime('%I:%M:%S %p'))
            start_time_str = st.selectbox("Start Time", time_options, index=start_time_index)
            end_time_index = time_options.index(df.index.max().strftime('%I:%M:%S %p'))
            end_time_str = st.selectbox("End Time", time_options, index=end_time_index)
            value_column = st.selectbox("Value Column", [col for col in df.columns if col != 'DateTime'])
            sampling_interval = st.slider("Sampling Interval (minutes)", 1, 60, 1)
            outlier_treatment = st.radio("Do you want to treat outliers?", ("No", "Yes"))
            
            degree = st.slider("Degree of Polynomial Regression", 1, 10, 2, key="degree_slider")
            annotation_text = st.text_input("Enter annotation text", key="annotation_text_input")
            annotation_x = st.text_input("Enter x value for annotation", key="annotation_x_input")
            annotation_y = st.text_input("Enter y value for annotation", key="annotation_y_input")
            if st.button("Add Annotation", key="add_annotation_button"):
                if 'annotations' not in st.session_state:
                    st.session_state.annotations = []
                st.session_state.annotations.append((annotation_text, annotation_x, annotation_y))
                st.experimental_rerun()
            forecast_periods = st.number_input("Forecasting Period (days)", min_value=1, max_value=365, value=180)    
            if st.button("Forecast Future", key="forecast_future_button"):
                st.session_state.forecast_future = True
                st.experimental_rerun()
            st.markdown("<div class='download-manual'>Download Manual</div>", unsafe_allow_html=True)
            download_manual()

    if 'df' in st.session_state:
        df = st.session_state.df

        st.markdown("<div class='df-overview-title'>DataFrame Overview</div>", unsafe_allow_html=True)
        df_display = df.head().copy()
        df_display.index = df_display.index.strftime('%Y-%m-%d %I:%M:%S %p')
        st.write(df_display)
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.markdown("<div class='df-overview-section'>Shape</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='df-shape-size'>{df.shape}</div>", unsafe_allow_html=True)
        st.markdown("<div class='df-overview-section'>Size</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='df-shape-size'>{df.size}</div>", unsafe_allow_html=True)

        for col in df.select_dtypes(include=['category', 'object']).columns:
            unique_values = df[col].unique()
            selected_values = st.multiselect(f"Filter by {col}", unique_values, default=unique_values)
            df = df[df[col].isin(selected_values)]

        start_datetime = datetime.strptime(f"{start_date} {start_time_str}", '%Y-%m-%d %I:%M:%S %p')
        end_datetime = datetime.strptime(f"{end_date} {end_time_str}", '%Y-%m-%d %I:%M:%S %p')
        mask = (df.index >= start_datetime) & (df.index <= end_datetime)
        filtered_df = df.loc[mask]
        resampled_df = get_resampled_df(filtered_df, sampling_interval)

        if outlier_treatment == "Yes":
            treated_df = treat_outliers(resampled_df, value_column)
        else:
            treated_df = resampled_df

        isolation_forest = IsolationForest(contamination=0.05)
        anomalies = isolation_forest.fit_predict(treated_df[[value_column]])
        treated_df['Anomaly'] = anomalies

        fig = go.Figure()

        inactivity_mask = (treated_df[value_column].rolling('10min').max() - treated_df[value_column].rolling('10min').min()) <= 15
        active_df = treated_df[~inactivity_mask]
        inactive_df = treated_df[inactivity_mask]

        fig.add_trace(go.Scatter(x=active_df.index, y=active_df[value_column], mode='lines', line=dict(color='blue'), name='Active Periods', connectgaps=True))
        fig.add_trace(go.Scatter(x=inactive_df.index, y=inactive_df[value_column], mode='lines', line=dict(color='red'), name='Inactivity Periods', connectgaps=True))
        fig.add_trace(go.Scatter(x=treated_df[treated_df['Anomaly'] == -1].index, y=treated_df[treated_df['Anomaly'] == -1][value_column], mode='markers', name='Anomalies', marker=dict(color='orange')))

        X = np.array((treated_df.index - treated_df.index.min()).total_seconds()).reshape(-1, 1)
        y = treated_df[value_column].values
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)

        fig.add_trace(go.Scatter(x=treated_df.index, y=y_pred, mode='lines', line=dict(color='green', dash='dash'), name='Regression Line'))

        fig.update_layout(title='Time Series Data with Inactivity Periods, Anomalies, and Regression Line', xaxis_title='DateTime', yaxis_title=value_column)
        st.plotly_chart(fig)
        st.markdown("**The time series plot displays the data over time, with blue lines representing active periods, red lines indicating inactivity periods, and orange markers highlighting anomalies. The green dashed line shows the linear regression line, which helps identify the overall trend in the data.**")

        decomposition = seasonal_decompose(treated_df[value_column], model='additive', period=30)
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
        control_chart_fig.add_trace(go.Scatter(x=treated_df.index, y=treated_df[value_column], mode='lines', name='Load cell Value', line=dict(color='blue')))
        control_chart_fig.add_trace(go.Scatter(x=treated_df.index, y=treated_df[value_column].rolling(window=30).std(), mode='lines', name='Rolling Std', line=dict(color='orange'), yaxis='y2'))
        control_chart_fig.update_layout(title='Control Charts (X-bar and R charts)', xaxis_title='DateTime', yaxis=dict(title=value_column), yaxis2=dict(title='Standard Deviation', overlaying='y', side='right'))
        st.plotly_chart(control_chart_fig, use_container_width=True)
        st.markdown("**The control chart monitors the process stability over time. The X-bar chart shows the mean of the process, and the R chart displays the range of the process variation. These charts help identify any unusual variations in the process.**")

        kmeans = KMeans(n_clusters=3)
        treated_df['Cluster'] = kmeans.fit_predict(treated_df[[value_column]])
        num_clusters = len(set(treated_df['Cluster']))

        if num_clusters > 1:
            silhouette_avg = silhouette_score(treated_df[[value_column]], treated_df['Cluster'])
        else:
            silhouette_avg = 'N/A'

        cluster_fig = go.Figure()
        colors = ['blue', 'orange', 'green']
        for cluster in range(num_clusters):
            cluster_data = treated_df[treated_df['Cluster'] == cluster]
            cluster_fig.add_trace(go.Scatter(x=cluster_data.index, y=cluster_data[value_column], mode='markers', marker=dict(color=colors[cluster]), name=f'Cluster {cluster}'))

        cluster_fig.update_layout(title=f'KMeans Clustering (Silhouette Score: {silhouette_avg})', xaxis_title='DateTime', yaxis_title=value_column)
        st.plotly_chart(cluster_fig, use_container_width=True)
        st.markdown("**The clustering plot uses KMeans to group the data into clusters. Each color represents a different cluster, helping to identify patterns and similarities within the data. The silhouette score indicates how well the data points fit within their clusters, with higher values representing better clustering.**")

        stats = treated_df[value_column].describe(percentiles=[.25, .5, .75])

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

        st.markdown("<div class='additional-visualizations'>Additional Visualizations</div>", unsafe_allow_html=True)

        st.markdown("<div class='histogram'>Histogram</div>", unsafe_allow_html=True)
        fig_hist = go.Figure()
        colors = px.colors.qualitative.Plotly

        # Filter numeric columns and remove 'Anomaly' and 'Cluster' columns
        numeric_columns = treated_df.select_dtypes(include=[np.number]).columns
        filtered_numeric_columns = [col for col in numeric_columns if col not in ['Anomaly', 'Cluster']]

        for i, col in enumerate(filtered_numeric_columns):
            fig_hist.add_trace(go.Histogram(x=treated_df[col], name=col, marker=dict(color=colors[i % len(colors)]), opacity=0.75))

        fig_hist.update_layout(barmode='overlay', title='Histogram of Numeric Columns', xaxis_title='Value', yaxis_title='Count', legend=dict(x=1, y=1, traceorder='normal'), bargap=0.2)
        fig_hist.update_traces(opacity=0.75)
        st.plotly_chart(fig_hist, use_container_width=True)
        logging.info("Histogram generated.")
        st.markdown("**The histogram visualizes the distribution of the data for each numeric column in the dataset.**")

        st.markdown("<div class='pair-plot'>Pair Plot</div>", unsafe_allow_html=True)
        pair_plot_fig = sns.pairplot(treated_df[filtered_numeric_columns], diag_kind='kde')
        st.pyplot(pair_plot_fig)
        logging.info("Pair plot generated.")
        st.markdown("**The pair plot displays pairwise relationships in the dataset, showing scatter plots for each pair of features and histograms for individual features.**")

        st.markdown("<div class='correlation-heatmap'>Correlation Heatmap</div>", unsafe_allow_html=True)
        corr = treated_df[filtered_numeric_columns].corr()
        fig_heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index.values, y=corr.columns.values, colorscale='Viridis'))
        fig_heatmap.update_layout(title='Correlation Heatmap')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        logging.info("Correlation heatmap generated.")
        st.markdown("**The correlation heatmap displays the correlation coefficients between pairs of features in the dataset. The colors represent the strength of the correlations.**")

        if 'annotations' in st.session_state:
            for annotation_text, annotation_x, annotation_y in st.session_state.annotations:
                fig.add_trace(go.Scatter(x=[annotation_x], y=[annotation_y], mode='text', text=[annotation_text], name='Annotation'))
            st.plotly_chart(fig)

        degree = st.session_state.get("degree_slider", 2)
        if degree:
            poly_features = np.polyfit(X.flatten(), y, degree)
            poly_model = np.poly1d(poly_features)
            y_poly_pred = poly_model(X.flatten())
            fig.add_trace(go.Scatter(x=treated_df.index, y=y_poly_pred[:len(treated_df.index)], mode='lines', line=dict(color='purple', dash='dot'), name=f'Polynomial Regression (degree {degree})'))
            st.plotly_chart(fig)

        if st.session_state.get('forecast_future', False):
            with st.spinner('Forecasting...'):
                try:
                    forecast = generate_forecast(treated_df, value_column, forecast_periods)
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

if __name__ == "__main__":
    main()
