# -*- coding: utf-8 -*-
"""
Neat Dash Stock Dashboard with responsive figures and date-based X-axis
VS Code ready
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime as dt, timedelta

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Stock Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Stock Information Dashboard", style={
        'text-align': 'center', 'background-color': '#007BFF',
        'color': 'white', 'padding': '20px', 'border-radius': '10px'
    }),
    html.Div([
        # Left input panel
        html.Div([
            html.Label("Enter Company code:", style={'color': 'black'}),
            dcc.Input(id="input-symbol", type="text", value="",placeholder='Enter Company Symbol',
                      style={'margin-bottom': '20px', 'padding': '10px', 'border-radius': '5px', 'color': 'black'}),
            dcc.DatePickerRange(id='date-picker-range',
                                start_date=dt(2023,1,1),
                                end_date=dt(2023,12,31),
                                style={'margin-bottom': '10px'}),
            dcc.Input(id='forecast-days-input', type='number', value="",
                      placeholder='Enter days for forecast',
                      style={'margin-bottom': '10px', 'padding': '10px', 'border-radius': '5px'}),
            html.Button('Update Visualization', id='update-button',
                        style={'background-color': '#4CAF50', 'color': 'white', 'padding': '10px', 'border-radius': '5px'}),
        ], style={'width': '25%', 'float': 'left', 'padding': '20px', 'background-color': '#f0f0f0'}),

        # Right panel for graphs
        html.Div([
            html.Div(id="company-info", style={'margin-bottom': '20px', 'text-align': 'center'}),
            dcc.Graph(id="stock-price-plot", style={'width':'100%', 'height':'500px'}),
            dcc.Graph(id="stock-prediction-plot", style={'width':'100%', 'height':'500px'}),
        ], style={'width': '70%', 'float': 'left', 'padding': '20px', 'text-align': 'center'}),
    ], style={'display': 'flex'}),
])

# Callback for company info and stock price plot
@app.callback(
    [Output("company-info", "children"),
     Output("stock-price-plot", "figure")],
    [Input("update-button", "n_clicks")],
    [State("input-symbol", "value"),
     State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date")]
)
def update_stock_price(n_clicks, input_symbol, start_date, end_date):
    if not input_symbol or n_clicks is None:
        return [], {}

    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        stock_data = yf.download(input_symbol, start=start_date, end=end_date, auto_adjust=False)
        if stock_data.empty:
            return ["No data found for this ticker/date range."], {}

        # Fix multi-level columns if needed
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)

        # Add simple moving average
        stock_data['SMA'] = stock_data['Close'].rolling(window=20).mean()

        fig = px.line(stock_data, x=stock_data.index, y='Close', title=f"{input_symbol} Stock Price Over Time")
        fig.add_scatter(x=stock_data.index, y=stock_data['SMA'], mode='lines', name='SMA')
        fig.update_layout(xaxis_title="Date", yaxis_title="Price", autosize=True,
                          margin=dict(l=40,r=40,t=50,b=40))

        company = yf.Ticker(input_symbol)
        info = company.info
        company_name = info.get('shortName', input_symbol)

        info_layout = [
            html.H2(company_name, style={'font-weight': 'bold', 'font-size': '24px'}),
            html.P(f"Industry: {info.get('industry','')}", style={'font-weight': 'bold'}),
            html.P(f"Sector: {info.get('sector','')}", style={'font-weight': 'bold'}),
            html.P(f"Market Cap: {info.get('marketCap','')}", style={'font-weight': 'bold'}),
            html.P(f"Forward P/E: {info.get('forwardPE','')}", style={'font-weight': 'bold'}),
            html.P(f"Beta: {info.get('beta','')}", style={'font-weight': 'bold'}),
            html.P(f"Previous Close: {info.get('regularMarketPreviousClose','')}", style={'font-weight': 'bold'}),
            html.P(f"Recommendation: {info.get('recommendationKey','')}", style={'font-weight': 'bold'})
        ]

        return info_layout, fig
    except Exception as e:
        print(f"Error details: {e}")
        return ["Error fetching company data."], {}

# Callback for stock prediction
@app.callback(
    Output("stock-prediction-plot", "figure"),
    [Input("update-button", "n_clicks")],
    [State("input-symbol", "value"),
     State("forecast-days-input", "value"),
     State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date")]
)
def update_stock_prediction(n_clicks, input_symbol, forecast_days, start_date, end_date):
    if not input_symbol or n_clicks is None or forecast_days is None:
        return {}

    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        forecast_days = int(forecast_days)
        if forecast_days <= 0:
            return {}

        stock_data = yf.download(input_symbol, start=start_date, end=end_date, auto_adjust=False)
        if stock_data.empty:
            return {}

        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)

        closing_prices = stock_data['Close'].dropna()
        if len(closing_prices) < 2:
            return {}

        # Prepare data
        X = np.arange(1, len(closing_prices)+1).reshape(-1,1)
        y = closing_prices.values

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).flatten()

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=2)

        param_grid = {'C':[0.1,1,10,100],'epsilon':[0.01,0.1,0.2,0.5],'kernel':['rbf']}
        svr = SVR()
        grid = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        best_svr = grid.best_estimator_

        X_future = np.arange(len(X_scaled)+1, len(X_scaled)+forecast_days+1).reshape(-1,1)
        X_combined = np.concatenate((X_scaled, scaler_X.transform(X_future)))

        y_pred = best_svr.predict(X_combined)
        y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1,1)).flatten()

        # Create dates for X-axis
        last_date = stock_data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        all_dates = stock_data.index.append(future_dates)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=y, mode='lines', name='Actual Prices', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=all_dates, y=y_pred_original, mode='lines', name='Predicted Prices', line=dict(color='green')))
        fig.update_layout(title=f"{input_symbol} Stock Prediction ({forecast_days} days)",
                          xaxis_title="Date", yaxis_title="Price", autosize=True,
                          margin=dict(l=40,r=40,t=50,b=40))
        return fig

    except Exception as e:
        print(f"Error details: {e}")
        return {}

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8051)
