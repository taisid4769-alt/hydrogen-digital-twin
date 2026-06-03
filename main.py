import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 網頁基本設定與全域變數
# ==========================================
st.set_page_config(page_title="PEM 電解槽數位孿生系統", layout="wide")
st.title("🔋 PEM 電解槽 AI 數位孿生與群體智慧最佳化")

# 初始化 session_state 用於儲存訓練好的模型與縮放器
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False

# ==========================================
# 核心演算法：模型建構與 PSO (後端邏輯)
# ==========================================

@st.cache_data
def preprocess_data(df):
    """清洗上傳的數據並進行標準化"""
    # 根據物理意義過濾雜訊
    df_clean = df[(df['PS-I-MON'] > 0.1) & (df['H-P1'] > 0.5) & (df['T-ELY-CH1'] > 20)]
    
    current = df_clean['PS-I-MON'].values
    p_h2 = df_clean['H-P1'].values
    p_o2 = df_clean['O-P1'].values
    temp = df_clean['T-ELY-CH1'].values
    voltage = df_clean['CV-mean'].values
    
    X = np.column_stack((current, p_h2, p_o2, temp))
    y = voltage
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    return X_scaled, y, scaler_X, len(df_clean)

def build_and_train_mlp(X_train, y_train, epochs, batch_size):
    """訓練多層感知器代理模型"""
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(4,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    
    # 建立進度條
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 自訂 Callback 以更新 Streamlit 進度條
    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"訓練進度: {epoch+1}/{epochs} Epochs | Loss: {logs['loss']:.5f}")
            
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.2, 
        callbacks=[early_stop, StreamlitCallback()],
        verbose=0
    )
    return model

class PEM_PSO_Optimizer:
    def __init__(self, model, scaler_X, alpha, beta, num_particles=40, max_iter=50):
        self.model = model
        self.scaler_X = scaler_X
        self.alpha = alpha
        self.beta = beta
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = 0.7, 2.0, 2.0
        
        self.bounds = np.array([
            [1.0, 15.0],  # 電流
            [5.0, 40.0],  # 氫壓
            [5.0, 40.0],  # 氧壓
            [40.0, 82.0]  # 溫度
        ])
        
    def fitness_function(self, positions):
        scaled_positions = self.scaler_X.transform(positions)
        predicted_voltage = self.model.predict(scaled_positions, verbose=0).flatten()
        currents = positions[:, 0]
        yield_score = currents / self.bounds[0, 1] 
        efficiency_score = 1.48 / predicted_voltage 
        return self.alpha * yield_score + self.beta * efficiency_score

    def optimize(self):
        dim = 4
        positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num_particles, dim))
        velocities = np.zeros((self.num_particles, dim))
        
        pbest_positions = np.copy(positions)
        pbest_fitness = np.zeros(self.num_particles) - np.inf
        gbest_position = np.zeros(dim)
        gbest_fitness = -np.inf
        
        history = []

        progress_bar = st.progress(0)
        
        for t in range(self.max_iter):
            fitness = self.fitness_function(positions)
            
            for i in range(self.num_particles):
                if fitness[i] > pbest_fitness[i]:
                    pbest_fitness[i] = fitness[i]
                    pbest_positions[i] = positions[i]
                    
                if fitness[i] > gbest_fitness:
                    gbest_fitness = fitness[i]
                    gbest_position = positions[i].copy()
            
            history.append(gbest_fitness)
            
            r1, r2 = np.random.rand(2, self.num_particles, dim)
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (pbest_positions - positions) + 
                          self.c2 * r2 * (gbest_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, self.bounds[:, 0], self.bounds[:, 1])
            
            progress_bar.progress((t + 1) / self.max_iter)
                
        return gbest_position, gbest_fitness, history

# ==========================================
# 前端介面設計 (UI Layout)
# ==========================================

# 側邊欄：檔案上傳
with st.sidebar:
    st.header("📂 資料管理")
    uploaded_file = st.file_uploader("上傳實驗數據 (test4_subset.csv)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV 載入成功！")
    else:
        st.info("請先上傳 CSV 檔案以啟用系統功能。")
        df = None

# 主畫面：建立三個功能分頁
tab1, tab2, tab3 = st.tabs(["📊 數據與模型訓練", "🧠 PSO 智慧最佳化", "🎛️ 數位孿生手動模擬"])

# ----------------- 分頁 1：數據與模型訓練 -----------------
with tab1:
    st.header("Step 1: 建立數位代理模型")
    if df is not None:
        st.subheader("原始數據預覽")
        st.dataframe(df.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("訓練回合數 (Epochs)", min_value=10, max_value=500, value=50)
        with col2:
            batch_size = st.selectbox("批次大小 (Batch Size)", [64, 128, 256, 512], index=2)
            
        if st.button("🚀 開始訓練模型", type="primary"):
            X_scaled, y, scaler, valid_count = preprocess_data(df)
            st.write(f"有效數據筆數：{valid_count} 筆")
            
            with st.spinner("AI 正在學習電化學特性..."):
                model = build_and_train_mlp(X_scaled, y, epochs, batch_size)
                
                # 儲存至 Session State
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.is_trained = True
                st.success("✅ 模型訓練完成！現在可以前往其他分頁進行分析。")
    else:
        st.warning("請先由左側欄上傳數據。")

# ----------------- 分頁 2：PSO 智慧最佳化 -----------------
with tab2:
    st.header("Step 2: 群體智慧尋找最佳操作點")
    if not st.session_state.is_trained:
        st.warning("請先至「數據與模型訓練」分頁完成模型訓練。")
    else:
        st.markdown("調整下方的權重，決定你要偏重**產氫量**還是**能源效率**：")
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            alpha = st.slider("產氫量權重 (α)", 0.0, 1.0, 0.5, 0.1)
        with col_w2:
            beta = st.slider("能源效率權重 (β)", 0.0, 1.0, 0.5, 0.1)
            
        if st.button("🔍 開始 PSO 演算法尋優", type="primary"):
            optimizer = PEM_PSO_Optimizer(
                model=st.session_state.model, 
                scaler_X=st.session_state.scaler, 
                alpha=alpha, beta=beta
            )
            
            with st.spinner("鳥群正在多維空間中搜尋最佳解..."):
                best_params, best_fit, history = optimizer.optimize()
            
            st.success("✅ 最佳化收斂完成！")
            
            # 顯示最佳解
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("最佳電流密度", f"{best_params[0]:.2f} A")
            res_col2.metric("最佳氫氣壓力", f"{best_params[1]:.2f} bar")
            res_col3.metric("最佳氧氣壓力", f"{best_params[2]:.2f} bar")
            res_col4.metric("最佳操作溫度", f"{best_params[3]:.2f} °C")
            
            st.subheader("PSO 收斂歷程")
            st.line_chart(pd.DataFrame(history, columns=["Fitness Score"]))

# ----------------- 分頁 3：數位孿生手動模擬 -----------------
with tab3:
    st.header("Step 3: 數位孿生即時預測")
    if not st.session_state.is_trained:
        st.warning("請先至「數據與模型訓練」分頁完成模型訓練。")
    else:
        st.markdown("手動調整操作參數，觀察 AI 模型即時預測的電壓與效率：")
        
        sim_c1, sim_c2 = st.columns([1, 1])
        with sim_c1:
            sim_current = st.slider("操作電流 (A)", 1.0, 15.0, 5.0)
            sim_hp = st.slider("氫氣壓力 (bar)", 5.0, 40.0, 20.0)
            sim_op = st.slider("氧氣壓力 (bar)", 5.0, 40.0, 20.0)
            sim_temp = st.slider("操作溫度 (°C)", 40.0, 82.0, 70.0)
            
        with sim_c2:
            # 即時運算
            input_data = np.array([[sim_current, sim_hp, sim_op, sim_temp]])
            scaled_input = st.session_state.scaler.transform(input_data)
            pred_voltage = st.session_state.model.predict(scaled_input, verbose=0)[0][0]
            efficiency = (1.48 / pred_voltage) * 100
            
            st.markdown("### ⚡ 預測結果")
            st.metric("預測工作電壓 (CV-mean)", f"{pred_voltage:.4f} V")
            st.metric("系統熱中性效率", f"{efficiency:.2f} %")
            
            if efficiency < 70:
                st.error("警告：效率過低！請考慮降低電流或提高溫度。")
            elif efficiency > 85:
                st.success("優秀：系統目前處於高效率運作狀態！")
