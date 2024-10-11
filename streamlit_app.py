import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import norm
import requests
import re
import streamlit as st
import yfinance as yf

class BlackScholes:
    def __init__(self, r, s, k, t, sigma):
        self.r = r          # Risk-free rate
        self.k = k          # Strike price
        self.s = s          # Stock price
        self.t = t          # Time to expiration
        self.sigma = sigma  # Volatility

    def calculate_df(self):
        try:
            d1 = (np.log(self.s / self.k) + (self.r + 0.5 * self.sigma**2) * self.t) / (self.sigma * np.sqrt(self.t))
            d2 = d1 - self.sigma * np.sqrt(self.t)
            return d1, d2
        except ZeroDivisionError:
            raise ValueError("Enter time value greater than 0")

    def option(self, option_type='Call'):
        d1, d2 = self.calculate_df()
        option_type = option_type.capitalize()
        try:
            if option_type == "Call":
                price = (self.s * norm.cdf(d1)) - (self.k * np.exp(-self.r * self.t) * norm.cdf(d2))
            elif option_type == "Put":
                price = (self.k * np.exp(-self.r * self.t) * norm.cdf(-d2)) - (self.s * norm.cdf(-d1))
            else:
                raise ValueError('Invalid input. Please enter "Call" or "Put"')

            return round(price, 2)
        except Exception as e:
            raise RuntimeError(f"Error calculating option price: {e}")

    def greeks(self, option_type):
        d1, d2 = self.calculate_df()
    
        try:
            pdf_d1 = norm.pdf(d1)
            cdf_d1 = norm.cdf(d1)
            cdf_neg_d1 = norm.cdf(-d1)
            cdf_d2 = norm.cdf(d2)
            cdf_neg_d2 = norm.cdf(-d2)
            sqrt_T = np.sqrt(self.t)
            exp_neg_rt = np.exp(-self.r * self.t)

            gamma = pdf_d1 / (self.s * self.sigma * sqrt_T)
            vega = self.s * pdf_d1 * sqrt_T 
            if option_type == "Call":
                delta = cdf_d1
                theta = (-self.s * pdf_d1 * self.sigma / (2 * sqrt_T)) - (self.r * self.k * exp_neg_rt * cdf_d2)
                rho = self.k * self.t * exp_neg_rt * cdf_d2
            elif option_type == "Put":
                delta = -cdf_neg_d1
                theta = (-self.s * pdf_d1 * self.sigma / (2 * sqrt_T)) + (self.r * self.k * exp_neg_rt * cdf_neg_d2)
                rho = -self.k * self.t * exp_neg_rt * cdf_neg_d2
            else:
                raise ValueError("Invalid option type. Must be 'Call' or 'Put'.")
        
            return {
                'delta': round(delta, 3),
                'gamma': round(gamma, 6),
                'theta': round(theta / 365, 6),  # Convert theta to per-day format
                'vega': round(vega * 0.01, 6),  # Vega is multiplied by 0.01 to adjust for percentage format
                'rho': round(rho * 0.01, 6)     # Rho in percentage format
            }

        except ZeroDivisionError:
            return "Error: Division by zero encountered in Greek calculations."
        except ValueError as e:
            return f"Error: {e}"

    def greek_visualisation(self, option_type, greek):
        fig = go.Figure()
    
        line_color = '#FA7070' if option_type == 'Call' else '#799351'
        min_s = self.s * 0.92
        max_s = self.s * 1.09
        spot_values = np.linspace(min_s, max_s, 200)

        greek_values = [BlackScholes(self.r, s, self.k, self.t, self.sigma).greeks(option_type)[greek] for s in spot_values]
        current_greek_value = BlackScholes(self.r, self.s, self.k, self.t, self.sigma).greeks(option_type)[greek]
        fig.add_trace(go.Scatter(x=spot_values, y=greek_values, mode='lines', name=greek.capitalize(), line=dict(color=line_color, width=3)))
        fig.add_trace(go.Scatter(x=[self.s], y=[current_greek_value], mode='markers', name=f'Current {greek.capitalize()}', marker=dict(color='black', size=7)))
        fig.update_layout(title=f'{greek.capitalize()} vs Spot Price ({option_type})', xaxis_title='Spot Price', yaxis_title=greek.capitalize())
        
        return fig

    def monte_carlo_pricing(self, num_simulations=10000):
        Z = np.random.standard_normal(num_simulations)
        ST = self.s * np.exp((self.r - 0.5 * self.sigma**2) * self.t + self.sigma * np.sqrt(self.t) * Z)
        payoffs = np.maximum(ST - self.k, 0)  # Call options
        option_price = np.exp(-self.r * self.t) * np.mean(payoffs)
        return option_price

def monte_carlo_pricing_visualization(spot_price, strike_price, time_to_expiry, volatility, risk_free_rate, num_simulations=1000, num_steps=252):
    dt = time_to_expiry / num_steps
    asset_paths = np.zeros((num_steps, num_simulations))
    asset_paths[0] = spot_price

    for t in range(1, num_steps):
        rand = np.random.standard_normal(num_simulations)
        asset_paths[t] = asset_paths[t - 1] * np.exp((risk_free_rate - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * rand)

    fig = go.Figure()
    
    for i in range(num_simulations):
        fig.add_trace(go.Scatter(x=np.linspace(0, time_to_expiry, num_steps), y=asset_paths[:, i], mode='lines', line=dict(width=1)))
    
    fig.update_layout(title="Monte Carlo Simulation: Asset Price Paths", xaxis_title="Time (Years)", yaxis_title="Asset Price", showlegend=False)

    return fig

def fetch_nifty():
    nifty_data = yf.download('^NSEI', interval='1m', period='1d')
    nifty_latest = nifty_data['Close'][-1]  # Get the last close price
    return float(nifty_latest)  # Ensure it is returned as a float

def main():
    st.title("Black-Scholes Option Pricing and Greek Visualizations")

    nifty_price = fetch_nifty()
    st.sidebar.header("Inputs")
    strike_price = st.sidebar.number_input("Strike Price", value=25000)
    time_to_expiry = st.sidebar.number_input("Time to Expiry (Years)", value=1.0)
    
    option_type = st.selectbox("Option Type", ['Call', 'Put'])
    
    spot_price = st.sidebar.slider('Stock Price', min_value=0.0, max_value=30000.0, value=nifty_price, step=5.0)
    volatility = st.sidebar.slider('Volatility (%)', min_value=0.0, max_value=100.0, value=20.0, step=0.25)
    risk_free_rate = st.sidebar.slider('Risk Free Rate (%)', min_value=0.0, max_value=100.0, value=5.0, step=0.01)
    num_steps = st.sidebar.number_input("Number of Steps", value=252, min_value=1)

    bs_model = BlackScholes(r=risk_free_rate / 100, s=spot_price, k=strike_price, t=time_to_expiry, sigma=volatility / 100)

    option_price = bs_model.option(option_type)
    st.sidebar.write(f"Option Price: {option_price}")

    greek_types = ['delta', 'gamma', 'theta', 'vega', 'rho']
    for greek in greek_types:
        fig = bs_model.greek_visualisation(option_type, greek)
        st.plotly_chart(fig)

    if st.sidebar.button("Run Monte Carlo Simulation"):
        num_simulations = st.sidebar.number_input("Number of Simulations", value=1000, min_value=1000,max_value=1000)
        monte_carlo_price = bs_model.monte_carlo_pricing(num_simulations=int(num_simulations))
        st.sidebar.write(f"Monte Carlo Option Price: {monte_carlo_price}")
        simulation_fig = monte_carlo_pricing_visualization(spot_price, strike_price, time_to_expiry, volatility / 100, risk_free_rate / 100, num_simulations, num_steps)
        st.plotly_chart(simulation_fig)

if __name__ == "__main__":
    main()
