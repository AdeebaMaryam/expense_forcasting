import streamlit as st
import os, math, traceback, subprocess, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

# --- Utility to ensure package is installed ---
# IMPORTANT: For Streamlit Community Cloud deployment, list all packages
# (streamlit, pandas, numpy, scikit-learn, xgboost, plotly, prophet)
# in a 'requirements.txt' file instead of using this function.
def ensure_install(import_name, pip_name=None, version=None):
    try:
        module = __import__(import_name)
        return module
    except Exception:
        pkg = pip_name if pip_name else import_name
        if version:
            pkg = f"{pkg}=={version}"
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            module = __import__(import_name)
            return module
        except Exception as e:
            st.error(f"Failed installing {pkg}: {e}")
            raise

prophet_mod = ensure_install("prophet", "prophet")
Prophet = prophet_mod.Prophet
sklearn = ensure_install("sklearn", "scikit-learn")
xgb_mod = ensure_install("xgboost", "xgboost")
xgb = xgb_mod
plotly = ensure_install("plotly", "plotly")
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Expense Forecasting", layout="wide")
st.title("💰 Expense Forecasting")

# --- FILE UPLOADER MODIFICATION for Cloud Deployment ---
# Use st.session_state to persist the DataFrame
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None

uploaded = st.file_uploader("Upload expenses.csv", type=["csv"])

if uploaded is not None:
    # Read directly from the file object (FIXED for cloud deployment)
    try:
        raw_df = pd.read_csv(uploaded)
        st.session_state['raw_data'] = raw_df
        st.success("File uploaded and ready for analysis.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.session_state['raw_data'] = None
        
# --- Main Execution Block ---
if st.button("Run Forecast") and st.session_state.get('raw_data') is not None:
    try:
        # --- Constants ---
        DATE_COL = "date"
        AMOUNT_COL = "amount"
        POSSIBLE_CATEGORY_COLS = ["category","cat","type","expense_type"]
        CATEGORIES = ["fuel","food","tools","penalty","repair","subscription","rent"]
        AGG_FREQ = "D"
        FORECAST_HORIZON = 14
        LAG_DAYS = [7,14,30]
        ROLL_WINDOWS = [7,30]
        MIN_XGB_ROWS = 3
        UP_THRESHOLD_PCT = 5.0
        DOWN_THRESHOLD_PCT = -5.0

        # --- Helper Functions ---
        def ensure_date(df, date_col=DATE_COL):
            df = df.copy()
            if date_col not in df.columns:
                for c in ['Date','ds','timestamp','datetime']:
                    if c in df.columns:
                        df = df.rename(columns={c:date_col})
                        break
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            return df

        def find_category_col(df):
            for c in POSSIBLE_CATEGORY_COLS:
                if c in df.columns:
                    return c
            for c in df.columns:
                if 'cat' in c.lower() or 'type' in c.lower():
                    return c
            return None

        def add_period(dt, freq, periods=1):
            if freq.upper() == "D":
                return dt + pd.Timedelta(days=periods)
            if freq.upper() == "W":
                return dt + pd.Timedelta(weeks=periods)
            if freq.upper() == "M":
                return dt + pd.DateOffset(months=periods)
            return dt + pd.Timedelta(days=periods)

        def aggregate_by_category(df, category, date_col=DATE_COL, amount_col=AMOUNT_COL, freq=AGG_FREQ):
            d = df.copy()
            d = ensure_date(d, date_col)
            d = d.dropna(subset=[date_col, amount_col])
            mask = d['__cat_match__'] == category
            sub = d.loc[mask, [date_col, amount_col]].copy()
            if sub.empty:
                return pd.DataFrame(columns=[date_col, amount_col])
            sub = sub.set_index(date_col)
            agg = sub.resample(freq)[amount_col].sum().reset_index()
            return agg

        def add_lag_features(df, date_col=DATE_COL, target_col=AMOUNT_COL, lags=None, windows=None):
            df = df.copy().set_index(date_col).sort_index()
            if lags:
                for lag in lags:
                    df[f'lag_{lag}'] = df[target_col].shift(lag)
            if windows:
                for w in windows:
                    df[f'rolling_mean_{w}'] = df[target_col].shift(1).rolling(w).mean()
            df = df.reset_index()
            return df

        def pct_change(curr, prev):
            try:
                if prev == 0 or math.isnan(prev) or math.isinf(prev):
                    return np.nan
                return (curr - prev) / prev * 100.0
            except Exception:
                return np.nan

        def pretty_pct(x):
            if pd.isna(x): return ""
            sign = "+" if x>0 else ""
            return f"{sign}{x:.1f}%"

        def trend_label(pct):
            if pd.isna(pct):
                return "Unknown"
            if pct >= UP_THRESHOLD_PCT:
                return "UP"
            if pct <= DOWN_THRESHOLD_PCT:
                return "DOWN"
            return "STABLE"

        def english_message(label, pct, date_str):
            if label=="UP":
                return f"On {date_str}, expenses likely go **UP** ({pretty_pct(pct)}). Plan accordingly."
            if label=="DOWN":
                return f"On {date_str}, expenses likely go **DOWN** ({pretty_pct(pct)})."
            if label=="STABLE":
                return f"On {date_str}, expenses expected to stay **STABLE** ({pretty_pct(pct)})."
            return ""

        # --- Iterative forecast function for XGBoost ---
        def iterative_forecast(last_known_df, pipeline, periods=FORECAST_HORIZON, freq=AGG_FREQ, used_lags=None, used_windows=None):
            cur = last_known_df.copy().sort_values(DATE_COL).reset_index(drop=True)
            preds=[]
            # Extract feature columns from the trained XGB model (after preprocessing)
            feature_cols = [c for c in pipeline.named_steps['xgb'].get_booster().feature_names]

            for i in range(periods):
                base = pd.to_datetime(cur[DATE_COL].max())
                next_date = add_period(base, freq, 1)
                row_df = pd.DataFrame([{DATE_COL: next_date}])
                
                # Add time-based features
                row_df['year'] = row_df[DATE_COL].dt.year
                row_df['month'] = row_df[DATE_COL].dt.month
                row_df['day'] = row_df[DATE_COL].dt.day
                row_df['dayofweek'] = row_df[DATE_COL].dt.dayofweek
                row_df['is_month_start'] = row_df[DATE_COL].dt.is_month_start.astype(int)
                row_df['is_month_end'] = row_df[DATE_COL].dt.is_month_end.astype(int)
                
                tmp = cur.set_index(DATE_COL).sort_index()
                
                # Add lag features
                if used_lags:
                    for lag in used_lags:
                        col = f'lag_{lag}'
                        row_df[col] = tmp[AMOUNT_COL].iloc[-lag] if len(tmp) >= lag else np.nan

                # Add rolling mean features
                if used_windows:
                    for w in used_windows:
                        col = f'rolling_mean_{w}'
                        try:
                            # Calculate rolling mean on the existing data up to the last known point
                            row_df[col] = tmp[AMOUNT_COL].shift(1).rolling(w).mean().iloc[-1]
                        except Exception:
                            row_df[col] = np.nan
                            
                # Align columns and prepare for prediction (filling missing with 0/ffill)
                for c in feature_cols:
                     if c not in row_df.columns:
                         row_df[c] = 0
                
                X_row = row_df[feature_cols].ffill().fillna(0)
                pred = pipeline.predict(X_row)[0]
                preds.append({DATE_COL: next_date, 'xgb_pred': float(pred)})
                
                # Update 'current' data with the prediction for the next iteration
                cur = pd.concat([cur, pd.DataFrame({DATE_COL:[next_date], AMOUNT_COL:[pred]})], ignore_index=True)
            return pd.DataFrame(preds)


        # --- Main Logic Start ---
        raw = st.session_state['raw_data'].copy()

        # Rename amount column if needed
        if AMOUNT_COL not in raw.columns:
            for c in ['amount','amt','value','cost','price','expense']:
                if c in raw.columns:
                    AMOUNT_COL = c
                    break
            else:
                st.error("Could not find an amount column. Rename your amount column to 'amount' or similar.")
                st.stop()
        
        # Determine category column
        cat_col = find_category_col(raw)
        if cat_col is None:
            st.info("No category column found. Forecasting total expenses only.")
            raw['__cat_match__'] = 'total'
            categories_to_run = ['total']
        else:
            raw['__cat_match__'] = raw[cat_col].astype(str).str.strip().str.lower()
            categories_to_run = [c.lower() for c in CATEGORIES]
            st.info(f"Using category column '{cat_col}'. Will attempt forecasts for: {', '.join(categories_to_run)}")

        # --- Forecasting Loop ---
        for category in categories_to_run:
            st.write("---")
            st.subheader(f"Category: {category.upper()} 📊")
            
            # Aggregate historical data
            ts = aggregate_by_category(raw, category, date_col=DATE_COL, amount_col=AMOUNT_COL, freq=AGG_FREQ)
            if ts.empty or ts[AMOUNT_COL].sum()==0:
                st.info(f"No historical data for **{category.upper()}** — skipping forecast.")
                continue

            ts = ts.sort_values(DATE_COL).reset_index(drop=True)
            st.write(f"Historical ({len(ts)} points) — showing last 5 rows for {category}:")
            st.dataframe(ts.tail(5))

            last_hist_date = pd.to_datetime(ts[DATE_COL].max())
            last_hist_amount = float(ts[AMOUNT_COL].iloc[-1])

            # --- Prophet Model ---
            prophet_df = ts.rename(columns={DATE_COL:'ds', AMOUNT_COL:'y'})[['ds','y']].copy()
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, verbose=False) 
            prophet_trained = False
            prophet_future = pd.DataFrame()
            
            try:
                with st.spinner(f"Training Prophet for {category}..."):
                    m.fit(prophet_df)
                prophet_trained = True
                
                future = m.make_future_dataframe(periods=FORECAST_HORIZON, freq=AGG_FREQ)
                prophet_pred_full = m.predict(future)[['ds','yhat']].rename(columns={'ds':'date','yhat':'prophet_pred'})
                prophet_future = prophet_pred_full[prophet_pred_full['date'] > last_hist_date].head(FORECAST_HORIZON).reset_index(drop=True)
                
                # Fill missing dates if Prophet's future DataFrame is short
                if len(prophet_future) < FORECAST_HORIZON:
                    last = prophet_future['date'].max() if not prophet_future.empty else last_hist_date
                    add_dates=[]
                    while len(prophet_future) + len(add_dates) < FORECAST_HORIZON:
                        last = add_period(last, AGG_FREQ, 1)
                        add_dates.append(last)
                    extra = pd.DataFrame({'date': add_dates, 'prophet_pred':[np.nan]*len(add_dates)})
                    prophet_future = pd.concat([prophet_future, extra], ignore_index=True)

            except Exception as e:
                st.warning(f"Prophet failed to fit for this category.")
                prophet_trained = False

            # --- XGBoost Model ---
            full_ts = ts[[DATE_COL, AMOUNT_COL]].copy()
            full_ts[DATE_COL] = pd.to_datetime(full_ts[DATE_COL])
            available_points = len(full_ts)
            usable_lags = [lag for lag in LAG_DAYS if lag < available_points]
            usable_windows = [w for w in ROLL_WINDOWS if w < available_points]

            # Add lag and rolling features
            feat = add_lag_features(full_ts, date_col=DATE_COL, target_col=AMOUNT_COL,
                                     lags=usable_lags if usable_lags else None,
                                     windows=usable_windows if usable_windows else None)
            
            # Add time-based features for XGBoost
            feat['year'] = feat[DATE_COL].dt.year
            feat['month'] = feat[DATE_COL].dt.month
            feat['day'] = feat[DATE_COL].dt.day
            feat['dayofweek'] = feat[DATE_COL].dt.dayofweek
            feat['is_month_start'] = feat[DATE_COL].dt.is_month_start.astype(int)
            feat['is_month_end'] = feat[DATE_COL].dt.is_month_end.astype(int)

            for c in feat.columns:
                if c!=DATE_COL:
                    feat[c] = pd.to_numeric(feat[c], errors='coerce')
            
            # Prepare data for training
            feature_cols = [c for c in feat.columns if c not in [DATE_COL, AMOUNT_COL]]
            feat_clean = feat.dropna(subset=[AMOUNT_COL] + feature_cols).reset_index(drop=True)
            
            xgb_pred_df = pd.DataFrame()
            xgb_trained = False
            
            if len(feat_clean) >= MIN_XGB_ROWS:
                X = feat_clean[feature_cols]; y = feat_clean[AMOUNT_COL]
                
                # Preprocessor for Imputation
                preprocessor = ColumnTransformer(
                    transformers=[('num', SimpleImputer(strategy='median'), feature_cols)],
                    remainder='passthrough'
                )
                
                xgb_pipe = Pipeline([('preprocessor', preprocessor),
                                     ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.07, objective='reg:squarederror', random_state=42))])
                try:
                    with st.spinner(f"Training XGBoost for {category}..."):
                        xgb_pipe.fit(X,y)
                    xgb_trained = True

                    # Iterative forecast
                    last_window = full_ts.tail(max(available_points, 10)).copy()
                    xgb_pred_df = iterative_forecast(last_window, xgb_pipe, periods=FORECAST_HORIZON, freq=AGG_FREQ,
                                                    used_lags=usable_lags if usable_lags else None,
                                                    used_windows=usable_windows if usable_windows else None)
                except Exception:
                    xgb_trained = False
                    st.warning(f"XGBoost failed to train for this category.")

            # --- Combine Predictions (Ensemble) ---
            if not prophet_trained and not xgb_trained:
                st.error("Both Prophet and XGBoost failed to produce a forecast for this category.")
                continue

            # Start prediction DataFrame with Prophet results, or an empty one if Prophet failed
            if not prophet_future.empty:
                preds = prophet_future[['date','prophet_pred']].copy().reset_index(drop=True)
            elif not xgb_pred_df.empty:
                preds = pd.DataFrame({'date': xgb_pred_df['date'], 'prophet_pred': np.nan})
            else:
                st.error("Forecast data missing.")
                continue
                
            preds['xgb_pred'] = np.nan
            if xgb_trained and not xgb_pred_df.empty:
                preds = preds.merge(xgb_pred_df, on='date', how='left')
                
            preds['prophet_pred'] = pd.to_numeric(preds['prophet_pred'], errors='coerce')
            preds['prophet_pred'] = preds['prophet_pred'].ffill().bfill() # Fill NaNs for blending
            
            def pick_best(row):
                # Simple averaging of available predictions
                vals=[]
                if pd.notna(row.get('prophet_pred')): vals.append(row['prophet_pred'])
                if pd.notna(row.get('xgb_pred')): vals.append(row['xgb_pred'])
                return float(np.mean(vals)) if vals else np.nan
                
            preds['prediction'] = preds.apply(pick_best, axis=1)

            # --- Calculate Trends and Messages ---
            for i in range(len(preds)):
                if i==0:
                    preds.at[i,'prev_amount'] = last_hist_amount
                else:
                    preds.at[i,'prev_amount'] = preds.at[i-1,'prediction']

            preds['pct_change'] = preds.apply(lambda r: pct_change(r['prediction'], r['prev_amount']), axis=1)
            preds['trend'] = preds['pct_change'].apply(lambda x: trend_label(x))
            preds['message_en'] = preds.apply(lambda r: english_message(r['trend'], r['pct_change'], r['date'].strftime('%Y-%m-%d')), axis=1)

            # Final rounding for display
            preds['prediction'] = preds['prediction'].round(2)
            preds['pct_change'] = preds['pct_change'].round(1)
            preds['prev_amount'] = preds['prev_amount'].round(2)

            # --- Summary Calculation ---
            if preds['prediction'].notna().any():
                half = max(1, len(preds)//2)
                first_avg = preds['prediction'].iloc[:half].mean()
                second_avg = preds['prediction'].iloc[half:].mean()
                overall_pct = pct_change(second_avg, first_avg)
                overall_trend = trend_label(overall_pct)
            else:
                overall_trend = "Unknown"
                overall_pct = np.nan

            # Fix for Pylance/Robustness: Define spike/dip rows explicitly
            if preds['pct_change'].notna().any():
                spike_idx = preds['pct_change'].idxmax()
                dip_idx = preds['pct_change'].idxmin()
                spike_row = preds.loc[spike_idx]
                dip_row = preds.loc[dip_idx]
            else:
                spike_row = None
                dip_row = None
                
            # --- Display Summary ---
            st.markdown(f"**Overall Trend Summary for next {FORECAST_HORIZON} periods:** **{overall_trend}** ({pretty_pct(overall_pct)})")
            
            if spike_row is not None and not pd.isna(spike_row['pct_change']) and spike_row['trend'] == 'UP':
                st.markdown(f" • Largest expected increase: **{spike_row['date'].strftime('%d %b %Y')}** — {pretty_pct(spike_row['pct_change'])} 📈")
            
            if dip_row is not None and not pd.isna(dip_row['pct_change']) and dip_row['trend'] == 'DOWN':
                st.markdown(f" • Largest expected decrease: **{dip_row['date'].strftime('%d %b %Y')}** — {pretty_pct(dip_row['pct_change'])} 📉")
            
            # --- Display Dataframe ---
            df_display = preds[['date','prediction','pct_change','trend','message_en']].rename(columns={
                'date':'Date','prediction':'Predicted Amount','pct_change':'Change (%)','trend':'Trend','message_en':'Note'
            })
            st.dataframe(df_display.head(14))

            # --- Plotting ---
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Scatter(x=ts[DATE_COL], y=ts[AMOUNT_COL], mode='lines', name='Historical', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=preds['date'], y=preds['prediction'], mode='lines', name='Prediction', line=dict(color='orange')))
            
            # Add markers for future predictions colored by trend
            color_map = {'UP':'red','DOWN':'green','STABLE':'gray','Unknown':'black'}
            legend_added = set()
            for idx,row in preds.iterrows():
                tr = row['trend']
                # Use 'tr' for marker color, but check 'legend_added' for showlegend
                legend_label = f"Trend: {tr}" if tr not in legend_added else None
                showlegend = legend_label is not None
                if legend_label:
                    legend_added.add(tr)
                fig.add_trace(go.Scatter(x=[row['date']], y=[row['prediction']], mode='markers',
                                         marker=dict(color=color_map.get(tr,'black'), size=8),
                                         name=legend_label, showlegend=showlegend))
                                         
            fig.update_layout(title=f"Expense Forecast for '{category}' (Historical & Future)",
                              xaxis_title="Date", yaxis_title="Expense Amount", height=500)
            st.plotly_chart(fig, use_container_width=True)

        # --- End of Try Block Success Message ---
        st.success("✅ All forecasts completed.")
        
    except Exception:
        # --- Catch-all Error Handling ---
        st.error("An error occurred while running the forecast.")
        st.code(traceback.format_exc())

# --- Warning for missing file upload ---
elif st.button("Run Forecast") and st.session_state.get('raw_data') is None:
    st.warning("Please upload a CSV file before running the forecast.")