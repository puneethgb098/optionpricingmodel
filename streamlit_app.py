import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import plotly.graph_objs as go
import seaborn as sns
from scipy.stats import norm
import requests
import re
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.stats import lognorm
from scipy.stats import gaussian_kde

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
    
    def greek_summary(r, S, K, T, sigma):
        model = BlackScholes(r, S, K, T, sigma)
        greeks_list = ['delta', 'gamma', 'vega', 'theta', 'rho']
        call_greeks = model.greeks('Call')
        put_greeks = model.greeks('Put')
    
        summary = {'Call Greeks': [call_greeks[greek] for greek in greeks_list],'Put Greeks': [put_greeks[greek] for greek in greeks_list]}
        df = pd.DataFrame(summary, index=['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])
    
        return df

    def monte_carlo_pricing(self, num_simulations=10000,option_type ='call'):
        Z = np.random.standard_normal(num_simulations)
        ST = self.s * np.exp((self.r - 0.5 * self.sigma**2) * self.t + self.sigma * np.sqrt(self.t) * Z)
        if option_type == 'call':
            payoffs = np.maximum(ST - self.k, 0) 
        elif option_type == 'put':
            payoffs = np.maximum(self.k - ST, 0)  
        else:
            raise ValueError("option_type should be 'call' or 'put'")
        option_price = np.exp(-self.r * self.t) * np.mean(payoffs)
        return option_price

    def american_option_pricing(self, s, k, t, r, n, sigma, option_type='call'):
        n = int(n)
        dt = t / n
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u                       # Down factor
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

        stock_price = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1):
                stock_price[j, i] = s * (u ** j) * (d ** (i - j))

        option_value = np.zeros((n + 1, n + 1))

        if option_type == 'call':
            for i in range(n + 1):
                option_value[i, n] = max(0, stock_price[i, n] - k)
        elif option_type == 'put':
            for i in range(n + 1):
                option_value[i, n] = max(0, k - stock_price[i, n])
        else:
            raise ValueError("option_type should be 'call' or 'put'")

        for j in range(n - 1, -1, -1):
            for i in range(j + 1):
                continuation_value = np.exp(-r * dt) * (p * option_value[i, j + 1] + (1 - p) * option_value[i + 1, j + 1])

                if option_type == 'call':
                    exercise_value = stock_price[i, j] - k
                elif option_type == 'put':
                    exercise_value = k - stock_price[i, j]

                option_value[i, j] = max(continuation_value, exercise_value)

        return option_value[0, 0]


def monte_carlo_pricing_visualization(option_value, strike_price, time_to_expiry, volatility, risk_free_rate, num_simulations=1000, num_steps=252):
    dt = time_to_expiry / num_steps 
    asset_paths = np.zeros((num_steps, num_simulations)) 
    asset_paths[0] = option_value  

    # Simulate asset price paths
    for t in range(1, num_steps):
        rand = np.random.standard_normal(num_simulations)  
        asset_paths[t] = asset_paths[t - 1] * np.exp((risk_free_rate - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * rand)

    final_prices = asset_paths[-1, :]

    kde = gaussian_kde(final_prices)
    price_range = np.linspace(min(final_prices), max(final_prices), 1000)
    pdf_values = kde(price_range)

    
    fig = make_subplots(
    rows=1, cols=2,  
    subplot_titles=("Asset Price Paths", "PDF of Final Asset Prices"),
    column_widths=[0.7, 0.3]
)

    for i in range(num_simulations):
        fig.add_trace(
        go.Scatter(
            x=np.linspace(0, time_to_expiry, num_steps),
            y=asset_paths[:, i],
            mode='lines',
            line=dict(width=1),
            showlegend=False
        ),
        row=1, col=1
    )

    fig.add_trace(
    go.Histogram(
        y=final_prices, 
        nbinsy=50,
        histnorm='probability',
        showlegend=False,
        opacity=0.6,
        marker=dict(color='blue'),
    ),
    row=1, col=2
)

    fig.update_layout(
    title="Monte Carlo Simulation and PDF of Final Asset Prices",
    template="plotly_white",
    xaxis_title="Time to Expiry",
    yaxis_title="Asset Price",
    xaxis2_title="Probability",
    yaxis2_title="Final Asset Prices",
    bargap=0.2

)
    return fig

def binomial_pricing_visualization(spot_price, strike_price, time_to_expiry, volatility, risk_free_rate, num_steps, option_type='Call'):
    dt = time_to_expiry / num_steps  
    u = np.exp(volatility * np.sqrt(dt))  
    d = 1 / u                            
    p = (np.exp(risk_free_rate * dt) - d) / (u - d)  

    # Calculate asset prices and option values at each step
    asset_prices = np.zeros((num_steps + 1, num_steps + 1))
    for i in range(num_steps + 1):
        for j in range(i + 1):
            asset_prices[j, i] = spot_price * (u ** (i - j)) * (d ** j)

    option_values = np.zeros_like(asset_prices)
    for i in range(num_steps + 1):
        if option_type.lower() == 'call':
            option_values[i, num_steps] = max(0, asset_prices[i, num_steps] - strike_price)
        elif option_type.lower() == 'put':
            option_values[i, num_steps] = max(0, strike_price - asset_prices[i, num_steps])

    # Backward induction to calculate option values
    for i in range(num_steps - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1]) * np.exp(-risk_free_rate * dt)
            if option_type.lower() == 'call':
                exercise_value = max(0, asset_prices[j, i] - strike_price)
            elif option_type.lower() == 'put':
                exercise_value = max(0, strike_price - asset_prices[j, i])
            option_values[j, i] = max(continuation_value, exercise_value)

    # Statistical insights: Mean and variance of final prices
    final_prices = asset_prices[:, num_steps]
    mean_price = np.mean(final_prices)
    variance_price = np.var(final_prices)

    # Visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Binomial Asset Tree", "Option Value Tree"),
        column_widths=[0.6, 0.4]
    )

    # Plot asset prices tree
    for i in range(num_steps + 1):
        fig.add_trace(
            go.Scatter(
                x=[i] * (i + 1),
                y=asset_prices[:i + 1, i],
                mode='markers+lines',
                name=f"Step {i}",
                marker=dict(size=6),
                line=dict(dash='dot')
            ),
            row=1, col=1
        )

    # Plot option values tree
    for i in range(num_steps + 1):
        fig.add_trace(
            go.Scatter(
                x=[i] * (i + 1),
                y=option_values[:i + 1, i],
                mode='markers+lines',
                name=f"Step {i}",
                marker=dict(size=6),
                line=dict(dash='dot', color='green')
            ),
            row=1, col=2
        )

    fig.update_layout(
        title=f"Binomial Pricing Visualization ({option_type.capitalize()} Option)",
        xaxis_title="Steps",
        yaxis_title="Asset/Option Value",
        template="plotly_dark"
    )

    # Add mean and variance as annotations
    fig.add_annotation(
        text=f"Mean Final Price: {mean_price:.2f}<br>Variance: {variance_price:.2f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=12, color="white"),
        align="center"
    )

    return fig


def fetch_nifty():
    try:
        nifty_latest = yf.download('^NSEI', interval = '1m', period = '1d')
        nifty_latest = round(nifty_latest.Close[-1], 1)
        return nifty_latest
    except:
        return 25000.0

def main():
    nifty_price = fetch_nifty()

    strike_price = 25000.0
    time_to_expiry = 1.0
    option_type = 'Call'
    spot_price = nifty_price
    volatility = 20.0
    risk_free_rate = 5.0
    num_steps=10.0

    option = st.sidebar.selectbox("Pick a strategy", ['Black Scholes Pricing', 'Monte Carlo Simulation', 'Binomial Tree Forecasting'], key='option_strategy')

    if option == 'Black Scholes Pricing':
        st.sidebar.header("Inputs")
        spot_price = st.sidebar.number_input('Stock Price', min_value=1.0, max_value=40000.0, value=nifty_price, step=5.0, key='spot_price')
        strike_price = st.sidebar.number_input("Strike Price", value=25000.0, min_value=1.0, max_value=40000.0, key='strike_price')
        time_to_expiry = st.sidebar.number_input("Time to Expiry (Years)", value=1.0, key='time_to_expiry')
        volatility = st.sidebar.number_input('Volatility (%)', min_value=1.0, max_value=100.0, value=20.0, step=0.25, key='volatility')
        risk_free_rate = st.sidebar.number_input('Risk Free Rate (%)', min_value=0.0, max_value=20.0, value=5.0, step=0.01, key='risk_free_rate')

        bs_model = BlackScholes(
            r=risk_free_rate / 100, 
            s=spot_price, 
            k=strike_price, 
            t=time_to_expiry, 
            sigma=volatility / 100)
    
        with st.container(border=True):
            st.write("### Option Price")

            call_price = bs_model.option("Call")
            put_price = bs_model.option("Put")

            call_greeks = bs_model.greeks("Call")
            put_greeks = bs_model.greeks("Put")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Call Option Price", value=f"{call_price:.4f}")
            with col2:
                st.metric(label="Put Option Price", value=f"{put_price:.4f}")
        
        with st.container(border=True):
            st.write("### Greek Values")
            greek_table = pd.DataFrame([[f"{call_greeks['delta']:.4f}", f"{call_greeks['gamma']:.4f}", f"{call_greeks['theta']:.4f}", f"{call_greeks['vega']:.4f}", f"{call_greeks['rho']:.4f}"],[f"{put_greeks['delta']:.4f}", f"{put_greeks['gamma']:.4f}", f"{put_greeks['theta']:.4f}", f"{put_greeks['vega']:.4f}", f"{put_greeks['rho']:.4f}"]],
            index=['Call Greeks', 'Put Greeks'],
            columns=['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
        )
        
            st.table(greek_table)
            
        with st.container(border=True):
            st.write("### Greek Visualizations")
            greek_types = ['delta', 'gamma', 'theta', 'vega', 'rho']

            col1, col2 = st.columns(2)
        
            for greek in greek_types:
                with col1:
                    call_fig = bs_model.greek_visualisation("Call", greek)
                    st.plotly_chart(call_fig, use_container_width=True)
            
                with col2:
                    put_fig = bs_model.greek_visualisation("Put", greek)
                    st.plotly_chart(put_fig, use_container_width=True)

        if st.button("Explain Black Scholes and Greeks"):
           st.write("""
    The Black-Scholes model is a mathematical model used to price European call and put options. 
    It assumes:
    - The option can only be exercised at expiration (European-style).
    - The price of the underlying asset follows a geometric Brownian motion.
    - Constant volatility and interest rate.

    The formula for a European option price is:
    \\
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    \\
    put_price = put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    \\
    \\
    where:
    - \\( S0 \\): Current stock price
    - \\( X \\): Strike price
    - \\( T \\): Time to expiration
    - \\( r \\): Risk-free interest rate
    - \\( sigma \\): Volatility of the underlying asset
    - \\( N(d) \\): Cumulative distribution function of the standard normal distribution
    - \\( d1 \\) = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    - \\( d2 \\) = d1 - sigma * np.sqrt(T)

    ## Option Greeks
    Option Greeks measure the sensitivity of an option's price to various factors:
    - **Delta (Δ):** Sensitivity to changes in the price of the underlying asset.
    - **Gamma (Γ):** Sensitivity of Delta to changes in the price of the underlying asset.
    - **Theta (Θ):** Sensitivity to the passage of time (time decay).
    - **Vega (ν):** Sensitivity to changes in volatility.
    - **Rho (ρ):** Sensitivity to changes in the risk-free interest rate.

    ### Example Use Cases:
    - Use Delta to hedge your portfolio.
    - Monitor Vega for options in volatile markets.
    - Understand Theta to manage time decay in your strategies.

    **Note:** This app provides tools to calculate and visualize these metrics.
    """)

    elif option == 'Monte Carlo Simulation':
        st.title("Monte Carlo Simulation")
    
    # Sidebar Inputs
        st.sidebar.header("Inputs")
        num_steps = st.sidebar.number_input("Number of Steps", value=252, min_value=1, key='num_steps_mc')
        num_simulations = st.sidebar.number_input("Number of Simulations", value=1000, min_value=500, max_value=2000, step=100, key='num_simulations')
        spot_price = st.sidebar.number_input('Stock Price', min_value=1.0, max_value=40000.0, value=nifty_price, step=5.0, key='spot_price')
        strike_price = st.sidebar.number_input("Strike Price", value=25000.0, min_value=1.0, max_value=40000.0, key='strike_price')
        time_to_expiry = st.sidebar.number_input("Time to Expiry (Years)", value=1.0, key='time_to_expiry')
        volatility = st.sidebar.number_input('Volatility (%)', min_value=1.0, max_value=100.0, value=20.0, step=0.25, key='volatility')
        risk_free_rate = st.sidebar.number_input('Risk Free Rate (%)', min_value=0.0, max_value=20.0, value=5.0, step=0.01, key='risk_free_rate')

        bs_model = BlackScholes(r=risk_free_rate / 100, s=spot_price, k=strike_price, t=time_to_expiry, sigma=volatility / 100)

        monte_carlo_call_price = bs_model.monte_carlo_pricing(num_simulations=int(num_simulations), option_type='call')
        monte_carlo_put_price = bs_model.monte_carlo_pricing(num_simulations=int(num_simulations), option_type='put')
        
        with st.container(border=True):
            col1, col2 = st.columns(2)

            with col1:
                st.metric(label="Call option Price:",value=f"{monte_carlo_call_price:.2f}")
    
            with col2:
                st.metric(label = "Put Option Price:",value=f"{monte_carlo_put_price:.2f}")
        
        with st.container(border=True):
            simulation_fig = monte_carlo_pricing_visualization(spot_price, strike_price, time_to_expiry, volatility / 100, risk_free_rate / 100, num_simulations, int(num_steps))
            st.plotly_chart(simulation_fig)

    else:
        st.sidebar.header("Inputs")
        st.title("Binomial Pricing for Nifty")

        option_type = st.selectbox("Option Type", ['Call', 'Put'], key='option_type')
        spot_price = st.sidebar.number_input("Stock Price", min_value=0.0, max_value=40000.0, value=24975.0, step=5.0)
        strike_price = st.sidebar.number_input("Strike Price", value=25000.0, min_value=1.0, max_value=40000.0, key='strike_price')
        time_to_expiry = st.sidebar.number_input("Time to Expiry (Years)", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
        volatility = st.sidebar.number_input("Volatility (%)", min_value=0.0, max_value=100.0, value=20.0) / 100
        risk_free_rate = st.sidebar.number_input("Risk Free Rate (%)", min_value=0.0, max_value=20.0, value=5.0) / 100
        num_steps = st.sidebar.number_input("Number of Steps", value=10, min_value=10, max_value=50, step=10)

        bs = BlackScholes(risk_free_rate, spot_price, strike_price, time_to_expiry, volatility)

        option_price = bs.american_option_pricing(spot_price, strike_price, time_to_expiry, risk_free_rate, num_steps, volatility, option_type.lower())
    
        st.write(f"The calculated option value is: **{option_price:.2f}**")

        if st.sidebar.button('Run'):
            binomial_tree_fig = binomial_pricing_visualization(spot_price, strike_price, time_to_expiry, volatility, risk_free_rate, num_steps, option_type)
            st.plotly_chart(binomial_tree_fig)

if __name__ == "__main__":
    main()

st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
col1, col2 = st.sidebar.columns(2)
col1.text("Linkedin:")
col1.page_link("https://www.linkedin.com/in/puneeth-g-b-463aa91a0/",label="Puneeth G B")
st.sidebar.text("")
st.sidebar.text("")
