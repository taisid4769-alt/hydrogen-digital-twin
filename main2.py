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
        # 加上 compile=False 避免反序列化錯誤
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

# ==========================================
# 核心演算法：PSO 最佳化類別
# ==========================================
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
        optimizer = PEM_PSO_Optimizer(
            model=model, 
            scaler_X=scaler, 
            alpha=alpha, 
            beta=beta
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

# ----------------- 分頁 2：數位孿生手動模擬 -----------------
with tab2:
    st.header("數位孿生即時預測")
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
        scaled_input = scaler.transform(input_data)
        pred_voltage = model.predict(scaled_input, verbose=0)[0][0]
        efficiency = (1.48 / pred_voltage) * 100

        st.markdown("### ⚡ 預測結果")
        st.metric("預測工作電壓 (CV-mean)", f"{pred_voltage:.4f} V")
        st.metric("系統熱中性效率", f"{efficiency:.2f} %")

        if efficiency < 70:
            st.error("警告：效率過低！請考慮降低電流或提高溫度。")
        elif efficiency > 85:
            st.success("優秀：系統目前處於高效率運作狀態！")
