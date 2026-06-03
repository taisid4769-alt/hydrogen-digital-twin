import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

st.set_page_config(page_title="PEM 電解槽數位孿生系統", layout="wide")
st.title("🔋 PEM 電解槽 AI 數位孿生與群體智慧最佳化")

# ==========================================
# 載入預訓練模型與縮放器 (快取起來，避免每次重整都讀取)
# ==========================================
@st.cache_resource
def load_ai_assets():
    model_path = 'pem_mlp_model.h5'
    scaler_path = 'scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    else:
        return None, None

model, scaler = load_ai_assets()

if model is None:
    st.error("❌ 找不到預訓練模型 (pem_mlp_model.h5) 或縮放器 (scaler.pkl)。請確認檔案已上傳！")
    st.stop() # 停止執行以下程式碼

st.success("✅ 已成功載入雲端預訓練 AI 代理模型！系統就緒。")

# --- (以下省略 PSO 的 class 定義，請把你們原本的 PEM_PSO_Optimizer 類別貼回這裡) ---
# class PEM_PSO_Optimizer:
#     ... (保持原樣)

# ==========================================
# 前端介面設計 (直接進入應用，不需訓練)
# ==========================================
tab1, tab2 = st.tabs(["🧠 PSO 智慧最佳化", "🎛️ 數位孿生手動模擬"])

# ----------------- 分頁 1：PSO 智慧最佳化 -----------------
with tab1:
    st.header("群體智慧尋找最佳操作點")
    st.markdown("調整下方的權重，決定你要偏重**產氫量**還是**能源效率**：")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        alpha = st.slider("產氫量權重 (α)", 0.0, 1.0, 0.5, 0.1)
    with col_w2:
        beta = st.slider("能源效率權重 (β)", 0.0, 1.0, 0.5, 0.1)

    if st.button("🔍 開始 PSO 演算法尋優", type="primary"):
        # 這裡直接使用我們載入好的 model 和 scaler
        optimizer = PEM_PSO_Optimizer(model=model, scaler_X=scaler, alpha=alpha, beta=beta)
        # ... (後續印出結果的程式碼保持原樣)
        
# ----------------- 分頁 2：數位孿生手動模擬 -----------------
with tab2:
    st.header("數位孿生即時預測")
    # ... (滑桿與預測的程式碼保持原樣，直接使用 model.predict() 即可)
