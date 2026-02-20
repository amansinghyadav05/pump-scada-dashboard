import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Industrial Pump SCADA", layout="wide")

st.title("⚙️ INDUSTRIAL PUMP SCADA CONTROL PANEL")

# Load data
df = pd.read_csv("pump_data.csv")

X = df[["Temperature", "Vibration"]]
y = df["Failure"]

model = LogisticRegression()
model.fit(X, y)

# Sidebar controls
st.sidebar.header("🎛 SENSOR INPUT")
temp = st.sidebar.slider("Temperature (°C)", 40, 120, 70)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 12.0, 3.0)

prob = model.predict_proba([[temp, vibration]])[0][1] * 100

# Create gauge function
def gauge_chart(title, value, max_value, color_steps):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "white"},
            'steps': color_steps,
        }
    ))
    fig.update_layout(height=300)
    return fig

col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(gauge_chart(
        "Temperature (°C)",
        temp,
        120,
        [
            {'range': [0, 70], 'color': "green"},
            {'range': [70, 90], 'color': "yellow"},
            {'range': [90, 120], 'color': "red"},
        ]
    ))

with col2:
    st.plotly_chart(gauge_chart(
        "Vibration (mm/s)",
        vibration,
        12,
        [
            {'range': [0, 4], 'color': "green"},
            {'range': [4, 8], 'color': "yellow"},
            {'range': [8, 12], 'color': "red"},
        ]
    ))

with col3:
    st.plotly_chart(gauge_chart(
        "Failure Risk (%)",
        prob,
        100,
        [
            {'range': [0, 40], 'color': "green"},
            {'range': [40, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "red"},
        ]
    ))

st.write("---")

# Alarm logic
if prob > 70:
    st.error("🔴 CRITICAL ALARM: Immediate shutdown required!")
elif prob > 40:
    st.warning("🟡 WARNING: Inspect pump condition.")
else:
    st.success("🟢 NORMAL: Pump operating safely.")

st.write("### 📊 Historical Data Trend")
st.line_chart(df[["Temperature", "Vibration"]])