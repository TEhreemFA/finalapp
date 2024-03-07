#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit app title
st.title('Influencer Engagement Analysis and Recommendations')
st.write('This application analyzes social media influencer profiles and provides recommendations.')

# File uploader widget
uploaded_file = st.file_uploader("Upload your influencer metrics CSV file", type=['csv'])

if uploaded_file is not None:
    # Reading the uploaded file into a DataFrame
    metrics_df = pd.read_csv(uploaded_file)
    metrics_df['Date'] = pd.to_datetime(metrics_df['Date'])  # Ensure 'Date' is the correct datetime format
    
    # Option to display raw data
    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.write(metrics_df)

    # Plotting Video Views over time
    st.subheader('Video Views Over Time')
    st.line_chart(metrics_df.set_index('Date')['Video views'])

    # Interactive Analysis: Likes and Comments Over Time
    st.subheader('Engagement Over Time')
    option = st.selectbox(
        'Choose metric to view over time:',
        ['Likes', 'Comments', 'Shares']
    )
    st.line_chart(metrics_df.set_index('Date')[option])

    # Weekly aggregation example
    st.subheader('Weekly Video views')
    weekly_data = metrics_df.resample('W', on='Date').sum()
    st.line_chart(weekly_data['Video views'])

    # Filtering data by date range selected by the user
    st.subheader('Filter by Date Range')
    start_date = st.date_input('Start date', value=metrics_df['Date'].min())
    end_date = st.date_input('End date', value=metrics_df['Date'].max())

    # Convert start_date and end_date to pandas datetime format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Now perform the filtering
    filtered_df = metrics_df[(metrics_df['Date'] >= start_date) & (metrics_df['Date'] <= end_date)]
    st.line_chart(filtered_df.set_index('Date')['Video views'])

    # Scatter plot for selected metrics
    st.subheader('Scatter Plot Analysis')
    metric_x = st.selectbox('Select the first metric:', metrics_df.columns[2:], index=1)  # Adjust the index based on your data
    metric_y = st.selectbox('Select the second metric:', metrics_df.columns[2:], index=2)  # Adjust the index based on your data
    fig, ax = plt.subplots()
    ax.scatter(filtered_df[metric_x], filtered_df[metric_y])
    plt.xlabel(metric_x)
    plt.ylabel(metric_y)
    st.pyplot(fig)

    # Recommendations
    st.subheader('Recommendations')
    st.write('Based on the trends observed in your data, consider diversifying your content to increase engagement across different types of videos. Engaging more with your audience through comments and community posts may also help maintain a high level of interaction.')

    # Best posting time suggestion based on the uploaded dataset
    st.subheader('Best Posting Time Suggestion')
    day_of_week = metrics_df['Date'].dt.day_name()
    avg_views_by_day = metrics_df.groupby(day_of_week)['Video views'].mean()
    best_day = avg_views_by_day.idxmax()
    st.write(f"Based on the analysis, posting on {best_day} might lead to higher video views.")

    best_post_time = '12:00 PM'  # Placeholder for dynamic calculation if needed
    st.write(f"Our analysis suggests that the best time to post for maximum engagement is around {best_post_time}.")
else:
    st.write("Please upload a dataset to analyze.")


# In[3]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Add your existing Streamlit code here...

# Predictive Analytics Section
st.header("Predictive Analytics")

if uploaded_file is not None and 'Video views' in metrics_df.columns and 'Likes' in metrics_df.columns:
    # Prepare the data for Linear Regression Model
    X = metrics_df[['Video views']]  # Independent variable
    y = metrics_df['Likes']  # Dependent variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Display the model's score
    score = model.score(X_test, y_test)
    st.write(f"Model Accuracy (R² score): {score:.2f}")

    # User input for scenario simulation
    st.subheader("Scenario Simulation")
    user_video_views = st.number_input("Enter your projected video views:", min_value=0, value=1000)
    
    # Predict likes based on user input
    predicted_likes = model.predict([[user_video_views]])[0]
    st.write(f"Predicted likes: {predicted_likes:.0f}")

    # Allow users to simulate different scenarios
    additional_views = st.slider("How many more views are expected?", 0, 10000, 500)
    updated_views = user_video_views + additional_views
    new_predicted_likes = model.predict([[updated_views]])[0]
    st.write(f"With an additional {additional_views} views, predicted likes could be: {new_predicted_likes:.0f}")

else:
    if uploaded_file:
        st.error("Your dataset must contain 'Video views' and 'Likes' columns for predictive analysis.")
    else:
        st.warning("Please upload a dataset to enable predictive analysis.")

