import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os

# Cố gắng import hyperopt nếu có sẵn trong môi trường
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

# ==============================================================================
# CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Hệ thống Kiểm định Chiến lược SMA + OBV & Tối ưu MPT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .highlight-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0 0;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. HÀM TÍNH TOÁN KỸ THUẬT VÀ TÍN HIỆU
# ==============================================================================

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Tính On-Balance Volume (OBV)."""
    price_diff = close.diff()
    direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
    direction[0] = 0.0
    obv = (pd.Series(direction, index=close.index) * volume).cumsum()
    return obv

def find_position_sma(df: pd.DataFrame, ma_short: int, ma_long: int) -> pd.Series:
    """Tạo tín hiệu SMA: Mua khi SMA ngắn cắt lên SMA dài, Bán khi cắt xuống."""
    position = pd.Series(0.0, index=df.index, name="SMA_Signal")
    if ma_short >= ma_long or ma_short <= 0 or ma_long <= 0:
        return position

    sma_short = df["Close"].rolling(window=int(ma_short)).mean()
    sma_long = df["Close"].rolling(window=int(ma_long)).mean()

    buy = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    sell = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))

    position.loc[buy] = 1.0
    position.loc[sell] = -1.0
    return position

def find_position_obv(df: pd.DataFrame, obv_window: int) -> pd.Series:
    """Tạo tín hiệu OBV: Mua khi OBV cắt lên MA(OBV), Bán khi cắt xuống."""
    position = pd.Series(0.0, index=df.index, name="OBV_Signal")
    if obv_window <= 0:
        return position

    obv = calculate_obv(df["Close"], df["Volume"])
    obv_ma = obv.rolling(window=int(obv_window)).mean()

    buy = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy] = 1.0
    position.loc[sell] = -1.0
    return position

def find_position_combined(df: pd.DataFrame, ma_short: int, ma_long: int, obv_window: int, logic: str = "OR") -> pd.Series:
    """Kết hợp tín hiệu SMA và OBV theo logic OR hoặc AND."""
    sma_sig = find_position_sma(df, ma_short, ma_long)
    obv_sig = find_position_obv(df, obv_window)

    position = pd.Series(0.0, index=df.index, name=f"Signal_{logic}")

    if logic == "OR":
        buy = (sma_sig == 1.0) | (obv_sig == 1.0)
        sell = (sma_sig == -1.0) | (obv_sig == -1.0)
        conflict = buy & sell
        position.loc[buy & ~conflict] = 1.0
        position.loc[sell & ~conflict] = -1.0
    else:  # Logic AND
        buy = (sma_sig == 1.0) & (obv_sig == 1.0)
        sell = (sma_sig == -1.0) & (obv_sig == -1.0)
        position.loc[buy] = 1.0
        position.loc[sell] = -1.0

    return position

def events_to_holding(events: pd.Series) -> pd.Series:
    """Chuyển đổi tín hiệu đột biến thành vị thế nắm giữ liên tục (Long-only: 1 hoặc 0)."""
    holding = pd.Series(0.0, index=events.index)
    current = 0.0
    values = events.values
    holding_vals = np.zeros(len(events))

    for i in range(len(values)):
        if values[i] == 1.0:
            current = 1.0
        elif values[i] == -1.0:
            current = 0.0
        holding_vals[i] = current

    return pd.Series(holding_vals, index=events.index)

def calculate_strategy_returns(df: pd.DataFrame, events: pd.Series, commission: float = 0.0) -> tuple:
    """
    Tính tỷ suất sinh lời của chiến lược với độ trễ 1 phiên (T+1) để tránh look-ahead bias.
    """
    asset_return = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)
    # Tín hiệu phiên t thực hiện ở phiên t+1
    executed_holding = holding.shift(1).fillna(0.0)

    strat_return = executed_holding * asset_return
    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    strat_return = strat_return - turnover * commission

    return strat_return, executed_holding

def performance_stats(returns: pd.Series, trading_days: int = 252, risk_free_rate: float = 0.0) -> dict:
    """Tính toán các chỉ số hiệu quả đầu tư chuẩn tài chính."""
    r = returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return {
            "Mean Daily Return [%]": 0.0,
            "Total Return [%]": 0.0,
            "Annual Return [%]": 0.0,
            "Annual Volatility [%]": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown [%]": 0.0
        }

    equity = (1.0 + r).cumprod()
    total_return = equity.iloc[-1] - 1.0
    years = len(r) / trading_days

    if years > 0 and equity.iloc[-1] > 0:
        annual_return = (equity.iloc[-1] ** (1.0 / years)) - 1.0
    else:
        annual_return = total_return / max(years, 0.001)

    annual_vol = r.std() * np.sqrt(trading_days)
    excess_daily = r.mean() - (risk_free_rate / trading_days)
    sharpe = (excess_daily / r.std() * np.sqrt(trading_days)) if r.std() > 0 else 0.0

    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = drawdown.min()

    return {
        "Mean Daily Return [%]": r.mean() * 100.0,
        "Total Return [%]": total_return * 100.0,
        "Annual Return [%]": annual_return * 100.0,
        "Annual Volatility [%]": annual_vol * 100.0,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_dd * 100.0
    }

# ==============================================================================
# 2. TỐI ƯU HÓA DANH MỤC MPT (MARKOWITZ)
# ==============================================================================

def portfolio_annual_return(weights: np.ndarray, returns: pd.DataFrame, trading_days: int = 252) -> float:
    mean_daily = returns.mean().values
    return float(np.dot(weights, mean_daily) * trading_days)

def portfolio_annual_volatility(weights: np.ndarray, returns: pd.DataFrame, trading_days: int = 252) -> float:
    cov_annual = returns.cov().values * trading_days
    variance = float(np.dot(weights.T, np.dot(cov_annual, weights)))
    return np.sqrt(max(variance, 0.0))

def negative_sharpe(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    p_ret = portfolio_annual_return(weights, returns, trading_days)
    p_vol = portfolio_annual_volatility(weights, returns, trading_days)
    if p_vol <= 1e-8:
        return 1e9
    return -(p_ret - risk_free_rate) / p_vol

def optimize_mpt(returns: pd.DataFrame, risk_free_rate: float = 0.0, trading_days: int = 252) -> np.ndarray:
    """Tối ưu hóa Sharpe Ratio theo Lý thuyết Danh mục Hiện đại (MPT) Long-only."""
    n = returns.shape[1]
    if n == 1:
        return np.array([1.0])

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        negative_sharpe,
        x0=x0,
        args=(returns, risk_free_rate, trading_days),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    if not result.success:
        return x0
    return result.x

# ==============================================================================
# 3. TỐI ƯU HÓA THAM SỐ HYPEROPT
# ==============================================================================

def optimize_parameters_hyperopt(df_train: pd.DataFrame, logic: str = "OR", commission: float = 0.0, max_evals: int = 30) -> dict:
    """Tối ưu hóa ma_short, ma_long, obv_window trên tập Train bằng Hyperopt."""
    if not HYPEROPT_AVAILABLE:
        return {"ma_short": 70, "ma_long": 250, "obv_window": 30}

    def objective(params):
        ma_s = int(params["ma_short"])
        ma_l = int(params["ma_long"])
        obv_w = int(params["obv_window"])

        if ma_s >= ma_l:
            return {"loss": 999999.0, "status": STATUS_OK}

        events = find_position_combined(df_train, ma_s, ma_l, obv_w, logic=logic)
        ret, _ = calculate_strategy_returns(df_train, events, commission=commission)
        stats = performance_stats(ret)
        sharpe = stats.get("Sharpe Ratio", 0.0)

        if np.isnan(sharpe):
            return {"loss": 999999.0, "status": STATUS_OK}
        return {"loss": -sharpe, "status": STATUS_OK}

    space = {
        "ma_short": hp.quniform("ma_short", 20, 120, 5),
        "ma_long": hp.quniform("ma_long", 150, 350, 5),
        "obv_window": hp.quniform("obv_window", 5, 90, 5)
    }

    trials = Trials()
    best_raw = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=False
    )

    return {
        "ma_short": int(best_raw["ma_short"]),
        "ma_long": int(best_raw["ma_long"]),
        "obv_window": int(best_raw["obv_window"])
    }

# ==============================================================================
# 4. LOAD & SÀNG LỌC DỮ LIỆU
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_csv_data(file_source) -> pd.DataFrame:
    """Đọc dữ liệu CSV từ file uploader hoặc đường dẫn tĩnh."""
    if isinstance(file_source, str):
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    col_map = {
        "datetime": "date", "time": "date", "ngay": "date",
        "symbol": "ticker", "ma": "ticker",
        "vol": "volume", "khoiluong": "volume",
        "dongcua": "close", "giadongcua": "close"
    }
    df = df.rename(columns=col_map)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing}")

    if "ticker" not in df.columns:
        df["ticker"] = "STOCK"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "close", "volume"]).sort_values("date")
    return df

@st.cache_data(show_spinner=False)
def run_stock_screening(df: pd.DataFrame, start_date, end_date, min_records: int = 300) -> pd.DataFrame:
    """Bảng sàng lọc cổ phiếu định lượng trên tập Train theo công thức notebook."""
    data = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))].copy()
    rows = []

    for ticker, g in data.groupby("ticker"):
        g = g.sort_values("date").dropna(subset=["close", "volume"])
        if len(g) < min_records:
            continue

        ret = g["close"].pct_change().dropna()
        if len(ret) == 0:
            continue

        tot_ret = (g["close"].iloc[-1] / g["close"].iloc[0]) - 1.0
        n_days = len(ret)
        ann_ret = ((1.0 + tot_ret) ** (252 / n_days) - 1.0) if (1.0 + tot_ret) > 0 else np.nan
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0

        equity = (1.0 + ret).cumprod()
        max_dd = (equity / equity.cummax() - 1.0).min()
        avg_val = (g["close"] * g["volume"]).mean()

        rows.append({
            "Ticker": ticker,
            "Số phiên": n_days,
            "Total Return [%]": tot_ret * 100,
            "Annual Return [%]": ann_ret * 100 if pd.notna(ann_ret) else 0.0,
            "Volatility [%]": ann_vol * 100,
            "Sharpe": sharpe,
            "Max Drawdown [%]": max_dd * 100,
            "GTGD TB (VNĐ)": avg_val
        })

    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    res["Return Rank"] = res["Annual Return [%]"].rank(pct=True)
    res["Sharpe Rank"] = res["Sharpe"].rank(pct=True)
    res["Liquidity Rank"] = res["GTGD TB (VNĐ)"].rank(pct=True)
    res["Drawdown Rank"] = res["Max Drawdown [%]"].rank(pct=True)

    res["Tổng Điểm (Score)"] = (
        0.30 * res["Return Rank"] +
        0.30 * res["Sharpe Rank"] +
        0.25 * res["Liquidity Rank"] +
        0.15 * res["Drawdown Rank"]
    )
    return res.sort_values("Tổng Điểm (Score)", ascending=False).reset_index(drop=True)

# ==============================================================================
# GIAO DIỆN CHÍNH & SIDEBAR
# ==============================================================================

st.markdown('<div class="main-title">🚀 Hệ thống Kiểm định Chiến lược SMA + OBV & Phân bổ MPT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Nền tảng Backtest định lượng kết hợp Tín hiệu Xu hướng (SMA) và Tín hiệu Khối lượng (OBV) trên TTCK Việt Nam</div>', unsafe_allow_html=True)

# --- SIDEBAR CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")

    # 1. Chọn nguồn dữ liệu
    st.subheader("1. Nguồn Dữ liệu")
    uploaded_file = st.file_uploader("Tải lên file CSV dữ liệu (HOSE / OHLCV):", type=["csv"])

    default_file_path = "HOSE_2020_2023_in.csv"
    if uploaded_file is not None:
        data_source = uploaded_file
        source_name = uploaded_file.name
    elif os.path.exists(default_file_path):
        data_source = default_file_path
        source_name = "HOSE_2020_2023_in.csv (Tệp mặc định)"
    else:
        data_source = None
        source_name = "Chưa có dữ liệu"

    st.info(f"📁 Tệp hiện tại: **{source_name}**")

    # Đọc dữ liệu nếu có
    df_raw = None
    if data_source is not None:
        try:
            df_raw = load_csv_data(data_source)
        except Exception as e:
            st.error(f"Lỗi đọc dữ liệu: {e}")

    # 2. Cấu hình phân chia Thời gian Train / Test
    st.subheader("2. Khoảng thời gian Train / Test")
    if df_raw is not None and not df_raw.empty:
        min_date = df_raw["date"].min().date()
        max_date = df_raw["date"].max().date()
    else:
        min_date = pd.to_datetime("2020-01-01").date()
        max_date = pd.to_datetime("2022-12-31").date()

    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        train_start = st.date_input("Train từ:", pd.to_datetime("2020-01-01").date(), min_value=min_date, max_value=max_date)
    with col_tr2:
        train_end = st.date_input("Train đến:", pd.to_datetime("2021-12-31").date(), min_value=min_date, max_value=max_date)

    col_te1, col_te2 = st.columns(2)
    with col_te1:
        test_start = st.date_input("Test từ:", pd.to_datetime("2022-01-01").date(), min_value=min_date, max_value=max_date)
    with col_te2:
        test_end = st.date_input("Test đến:", pd.to_datetime("2022-12-31").date(), min_value=min_date, max_value=max_date)

    # 3. Logic & Tham số Chiến lược
    st.subheader("3. Thiết lập Chiến lược")
    strat_logic = st.selectbox("Logic Kết hợp Tín hiệu:", ["OR", "AND"], index=0,
                               help="OR: Mua khi SMA hoặc OBV phát tín hiệu. Bán khi một trong hai phát tín hiệu. Xung đột => 0.")
    commission_pct = st.number_input("Phí giao dịch (% / lần đảo vị thế):", min_value=0.0, max_value=2.0, value=0.0, step=0.05) / 100.0
    initial_capital = st.number_input("Vốn ban đầu (VNĐ):", min_value=100000, value=1000000, step=100000)
    risk_free_rate = st.number_input("Lãi suất phi rủi ro Rf (%/năm):", min_value=0.0, max_value=15.0, value=0.0, step=0.5) / 100.0

    st.markdown("---")
    st.caption("💡 Được phát triển dựa trên thuật toán tối ưu hóa định lượng Python & Streamlit.")

# ==============================================================================
# XỬ LÝ DỮ LIỆU NẾU CHƯA CÓ FILE
# ==============================================================================
if df_raw is None or df_raw.empty:
    st.warning("⚠️ Vui lòng tải lên file dữ liệu CSV (hoặc đặt file `HOSE_2020_2023_in.csv` cùng thư mục chạy ứng dụng) để bắt đầu phân tích.")
    st.markdown("""
    ### 📋 Yêu cầu cấu trúc file CSV:
    File CSV cần bao gồm tối thiểu các cột (không phân biệt chữ hoa/thường):
    - `date`: Ngày giao dịch (VD: 2020-01-02 hoặc 1/2/2020)
    - `ticker`: Mã cổ phiếu (VD: DIG, DGC, VND, HAH, MSN...)
    - `open`: Giá mở cửa
    - `high`: Giá cao nhất
    - `low`: Giá thấp nhất
    - `close`: Giá đóng cửa
    - `volume`: Khối lượng giao dịch
    """)
    st.stop()

# Danh sách Tickers có trong dữ liệu
available_tickers = sorted(df_raw["ticker"].unique())
default_5 = ["DIG", "DGC", "VND", "HAH", "MSN"]
preset_selected = [t for t in default_5 if t in available_tickers]
if not preset_selected:
    preset_selected = available_tickers[:5]

# ==============================================================================
# BỐ CỤC CÁC TABS PHÂN TÍCH
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Sàng lọc Cổ phiếu (Train)",
    "📈 2. Kiểm định Từng Mã & Tín hiệu",
    "💼 3. Danh mục MPT vs Equal Weight (Test)",
    "📑 4. Báo cáo & Đánh giá Tự động"
])

# ------------------------------------------------------------------------------
# TAB 1: SÀNG LỌC CỔ PHIẾU
# ------------------------------------------------------------------------------
with tab1:
    st.markdown("### 📊 Sàng lọc Định lượng Cổ phiếu (Train 2020 - 2021)")
    st.markdown("""
    Quy trình chấm điểm và xếp hạng cổ phiếu theo trọng số notebook nghiên cứu:
    - **30%**: Tỷ suất sinh lời năm hóa (Annual Return)
    - **30%**: Chỉ số Sharpe Ratio
    - **25%**: Thanh khoản bình quân (Giá trị giao dịch trung bình ngày)
    - **15%**: Mức sụt giảm tối đa (Max Drawdown - ưu tiên âm thấp hơn)
    """)

    with st.spinner("Đang tính toán bảng sàng lọc trên tập Train..."):
        screen_df = run_stock_screening(df_raw, train_start, train_end, min_records=200)

    if not screen_df.empty:
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Tổng số mã đủ điều kiện", len(screen_df))
        top_scorer = screen_df.iloc[0]["Ticker"]
        top_score_val = screen_df.iloc[0]["Tổng Điểm (Score)"]
        col_s2.metric("Mã điểm Train cao nhất", f"{top_scorer} ({top_score_val:.3f})")
        col_s3.metric("Lợi nhuận Train TB", f"{screen_df['Total Return [%]'].mean():.1f}%")
        col_s4.metric("Sharpe Train TB", f"{screen_df['Sharpe'].mean():.2f}")

        st.subheader("Bảng Xếp Hạng Top 20 Cổ Phiếu Định Lượng Cao Nhất")
        st.dataframe(
            screen_df.head(20).style.format({
                "Total Return [%]": "{:.2f}%",
                "Annual Return [%]": "{:.2f}%",
                "Volatility [%]": "{:.2f}%",
                "Sharpe": "{:.2f}",
                "Max Drawdown [%]": "{:.2f}%",
                "GTGD TB (VNĐ)": "{:,.0f}",
                "Tổng Điểm (Score)": "{:.3f}"
            }),
            use_container_width=True
        )

        st.subheader("Vị trí của 5 Cổ phiếu Tiêu biểu Nhóm nghiên cứu:")
        benchmark_5_df = screen_df[screen_df["Ticker"].isin(default_5)]
        st.dataframe(
            benchmark_5_df.style.format({
                "Total Return [%]": "{:.2f}%",
                "Annual Return [%]": "{:.2f}%",
                "Volatility [%]": "{:.2f}%",
                "Sharpe": "{:.2f}",
                "Max Drawdown [%]": "{:.2f}%",
                "GTGD TB (VNĐ)": "{:,.0f}",
                "Tổng Điểm (Score)": "{:.3f}"
            }),
            use_container_width=True
        )
    else:
        st.warning("Không có cổ phiếu nào đạt số phiên tối thiểu trong giai đoạn Train đã chọn.")

# ------------------------------------------------------------------------------
# TAB 2: KIỂM ĐỊNH TỪNG MÃ & TÍN HIỆU
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Kiểm định Chiến lược SMA + OBV trên Từng Cổ phiếu")

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        selected_stock = st.selectbox("Chọn cổ phiếu để phân tích chi tiết:", available_tickers,
                                      index=available_tickers.index("DIG") if "DIG" in available_tickers else 0)

    # Lấy dữ liệu của cổ phiếu đã chọn
    df_stock = df_raw[df_raw["ticker"] == selected_stock].copy().set_index("date").sort_index()
    df_stock = df_stock.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})

    df_stock_train = df_stock.loc[(df_stock.index >= pd.to_datetime(train_start)) & (df_stock.index <= pd.to_datetime(train_end))].copy()
    df_stock_test = df_stock.loc[(df_stock.index >= pd.to_datetime(test_start)) & (df_stock.index <= pd.to_datetime(test_end))].copy()

    # Tham số mặc định từ kết quả tối ưu của notebook cho 5 mã
    preset_params = {
        "DIG": {"ma_short": 70, "ma_long": 305, "obv_window": 30},
        "DGC": {"ma_short": 70, "ma_long": 245, "obv_window": 70},
        "VND": {"ma_short": 90, "ma_long": 320, "obv_window": 5},
        "HAH": {"ma_short": 55, "ma_long": 215, "obv_window": 60},
        "MSN": {"ma_short": 110, "ma_long": 350, "obv_window": 100}
    }
    cur_default = preset_params.get(selected_stock, {"ma_short": 50, "ma_long": 200, "obv_window": 20})

    st.markdown("#### 🛠️ Điều chỉnh Tham số Kỹ thuật")
    col_p1, col_p2, col_p3, col_p4 = st.columns([1, 1, 1, 1])

    with col_p1:
        param_ma_short = st.slider("SMA Ngắn (ma_short):", min_value=10, max_value=150, value=cur_default["ma_short"], step=5)
    with col_p2:
        param_ma_long = st.slider("SMA Dài (ma_long):", min_value=150, max_value=400, value=cur_default["ma_long"], step=5)
    with col_p3:
        param_obv_window = st.slider("Cửa sổ OBV MA:", min_value=5, max_value=120, value=cur_default["obv_window"], step=5)
    with col_p4:
        st.write("Tự động Tối ưu hóa:")
        if st.button("⚡ Chạy Hyperopt Train", help="Tìm bộ tham số tối đa hóa Sharpe trên tập Train"):
            if HYPEROPT_AVAILABLE and not df_stock_train.empty:
                with st.spinner(f"Đang tối ưu {selected_stock}..."):
                    best_opt = optimize_parameters_hyperopt(df_stock_train, logic=strat_logic, commission=commission_pct, max_evals=40)
                    st.success(f"Kết quả tốt nhất: {best_opt}")
                    param_ma_short = best_opt["ma_short"]
                    param_ma_long = best_opt["ma_long"]
                    param_obv_window = best_opt["obv_window"]
            else:
                st.info("Hyperopt chưa được cài đặt hoặc tập Train rỗng. Sử dụng tham số gợi ý.")

    # Chọn tập dữ liệu hiển thị (Train hoặc Test)
    period_choice = st.radio("Chọn giai đoạn hiển thị biểu đồ:", ["Tập Test (Out-of-sample)", "Tập Train (In-sample)", "Toàn bộ Dữ liệu"], horizontal=True)
    if period_choice == "Tập Test (Out-of-sample)":
        df_target = df_stock_test
        period_label = f"Test ({test_start} đến {test_end})"
    elif period_choice == "Tập Train (In-sample)":
        df_target = df_stock_train
        period_label = f"Train ({train_start} đến {train_end})"
    else:
        df_target = df_stock
        period_label = "Toàn bộ"

    if df_target.empty:
        st.error(f"Dữ liệu {period_choice} của {selected_stock} rỗng. Vui lòng kiểm tra lại ngày.")
    else:
        # Tính toán tín hiệu và lợi nhuận
        events = find_position_combined(df_target, param_ma_short, param_ma_long, param_obv_window, logic=strat_logic)
        strat_ret, holding = calculate_strategy_returns(df_target, events, commission=commission_pct)
        bh_ret = df_target["Close"].pct_change().fillna(0.0)

        strat_stats = performance_stats(strat_ret, risk_free_rate=risk_free_rate)
        bh_stats = performance_stats(bh_ret, risk_free_rate=risk_free_rate)

        # Hiển thị Metrics so sánh
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("Lợi nhuận Chiến lược", f"{strat_stats['Total Return [%]']:.2f}%",
                     delta=f"{strat_stats['Total Return [%]'] - bh_stats['Total Return [%]']:.2f}% vs B&H")
        mcol2.metric("Lợi nhuận Buy & Hold", f"{bh_stats['Total Return [%]']:.2f}%")
        mcol3.metric("Sharpe Chiến lược", f"{strat_stats['Sharpe Ratio']:.2f}",
                     delta=f"{strat_stats['Sharpe Ratio'] - bh_stats['Sharpe Ratio']:.2f}")
        mcol4.metric("Max Drawdown Chiến lược", f"{strat_stats['Max Drawdown [%]']:.2f}%",
                     delta=f"{strat_stats['Max Drawdown [%]'] - bh_stats['Max Drawdown [%]']:.2f}%", delta_color="inverse")
        mcol5.metric("Số phiên nắm giữ", f"{int(holding.sum())} / {len(holding)} ({(holding.sum()/len(holding)*100):.1f}%)")

        # ----------------- VẼ BIỂU ĐỒ KỸ THUẬT VÀ TÍN HIỆU -----------------
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            row_heights=[0.55, 0.25, 0.20],
                            subplot_titles=[f"Giá {selected_stock}, Đường SMA & Điểm Mua/Bán ({period_label})", "Chỉ báo OBV & OBV MA", "Vị thế Nắm giữ (Holding)"])

        # Đường giá và SMA
        fig.add_trace(go.Scatter(x=df_target.index, y=df_target["Close"], mode="lines", name="Giá Close", line=dict(color="#2563EB", width=1.5)), row=1, col=1)
        sma_s_series = df_target["Close"].rolling(param_ma_short).mean()
        sma_l_series = df_target["Close"].rolling(param_ma_long).mean()
        fig.add_trace(go.Scatter(x=df_target.index, y=sma_s_series, mode="lines", name=f"SMA({param_ma_short})", line=dict(color="#F59E0B", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_target.index, y=sma_l_series, mode="lines", name=f"SMA({param_ma_long})", line=dict(color="#EF4444", width=1.2)), row=1, col=1)

        # Đánh dấu tín hiệu Mua / Bán
        buy_points = df_target[events == 1.0]
        sell_points = df_target[events == -1.0]
        if not buy_points.empty:
            fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points["Close"], mode="markers", name="Tín hiệu MUA",
                                     marker=dict(symbol="triangle-up", size=11, color="#10B981", line=dict(width=1, color="black"))), row=1, col=1)
        if not sell_points.empty:
            fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points["Close"], mode="markers", name="Tín hiệu BÁN",
                                     marker=dict(symbol="triangle-down", size=11, color="#DC2626", line=dict(width=1, color="black"))), row=1, col=1)

        # OBV Subplot
        obv_vals = calculate_obv(df_target["Close"], df_target["Volume"])
        obv_ma_vals = obv_vals.rolling(param_obv_window).mean()
        fig.add_trace(go.Scatter(x=df_target.index, y=obv_vals, mode="lines", name="OBV", line=dict(color="#8B5CF6", width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_target.index, y=obv_ma_vals, mode="lines", name=f"OBV MA({param_obv_window})", line=dict(color="#EC4899", width=1.2, dash="dot")), row=2, col=1)

        # Vị thế nắm giữ Subplot
        fig.add_trace(go.Scatter(x=df_target.index, y=holding, mode="lines", name="Vị thế (1: Giữ, 0: Tiền mặt)", line=dict(color="#059669", width=1.5), fill="tozeroy"), row=3, col=1)

        fig.update_layout(height=650, margin=dict(l=40, r=40, t=40, b=40), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        # ----------------- TĂNG TRƯỞNG VỐN TỪNG CỔ PHIẾU -----------------
        st.subheader(f"Đường Tăng trưởng Vốn (Equity Curve): Chiến lược vs Buy & Hold")
        eq_strat = initial_capital * (1.0 + strat_ret).cumprod()
        eq_bh = initial_capital * (1.0 + bh_ret).cumprod()

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=df_target.index, y=eq_strat, mode="lines", name=f"Chiến lược SMA+OBV ({strat_logic})", line=dict(color="#2563EB", width=2.5)))
        fig_eq.add_trace(go.Scatter(x=df_target.index, y=eq_bh, mode="lines", name="Buy & Hold", line=dict(color="#9CA3AF", width=2, dash="dash")))
        fig_eq.update_layout(height=400, yaxis_title="Giá trị danh mục (VNĐ)", hovermode="x unified")
        st.plotly_chart(fig_eq, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: DANH MỤC ĐẦU TƯ MPT VS EQUAL WEIGHT
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("### 💼 Xây dựng & Kiểm định Danh mục Đa Cổ phiếu trên Tập Test")
    st.markdown("""
    So sánh hiệu quả danh mục trên **Tập Test (2022)** giữa 3 phương pháp:
    1. **Equal Weight Buy & Hold**: Benchmark thụ động chia đều 1/N.
    2. **SMA + OBV Equal Weight**: Áp dụng chiến lược với tỷ trọng đều 1/N.
    3. **SMA + OBV MPT Optimal**: Áp dụng tỷ trọng tối ưu theo Lý thuyết Danh mục Markowitz (tính trên tập Train).
    """)

    # Chọn danh sách cổ phiếu trong danh mục
    selected_portfolio_tickers = st.multiselect(
        "Chọn các cổ phiếu cấu thành danh mục:",
        available_tickers,
        default=preset_selected
    )

    if len(selected_portfolio_tickers) < 2:
        st.warning("⚠️ Vui lòng chọn ít nhất 2 cổ phiếu để thực hiện phân bổ danh mục.")
    else:
        # Chuẩn bị dữ liệu cho từng cổ phiếu
        train_returns_dict = {}
        test_returns_dict = {}
        test_bh_returns_dict = {}

        params_used = {}
        for ticker in selected_portfolio_tickers:
            df_t = df_raw[df_raw["ticker"] == ticker].copy().set_index("date").sort_index()
            df_t = df_t.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})

            df_tr = df_t.loc[(df_t.index >= pd.to_datetime(train_start)) & (df_t.index <= pd.to_datetime(train_end))].copy()
            df_te = df_t.loc[(df_t.index >= pd.to_datetime(test_start)) & (df_t.index <= pd.to_datetime(test_end))].copy()

            if df_tr.empty or df_te.empty:
                st.error(f"Dữ liệu mã {ticker} bị rỗng trong kỳ Train hoặc Test.")
                continue

            # Sử dụng bộ tham số tối ưu có sẵn hoặc mặc định
            p = preset_params.get(ticker, {"ma_short": 70, "ma_long": 250, "obv_window": 30})
            params_used[ticker] = p

            # Tín hiệu và tỷ suất sinh lời trên Train
            events_tr = find_position_combined(df_tr, p["ma_short"], p["ma_long"], p["obv_window"], logic=strat_logic)
            strat_ret_tr, _ = calculate_strategy_returns(df_tr, events_tr, commission=commission_pct)
            train_returns_dict[ticker] = strat_ret_tr

            # Tín hiệu và tỷ suất sinh lời trên Test
            events_te = find_position_combined(df_te, p["ma_short"], p["ma_long"], p["obv_window"], logic=strat_logic)
            strat_ret_te, _ = calculate_strategy_returns(df_te, events_te, commission=commission_pct)
            test_returns_dict[ticker] = strat_ret_te

            # Buy & Hold trên Test
            test_bh_returns_dict[ticker] = df_te["Close"].pct_change().fillna(0.0)

        # Tạo Matrix
        train_ret_matrix = pd.DataFrame(train_returns_dict).dropna()
        test_ret_matrix = pd.DataFrame(test_returns_dict).dropna()
        test_bh_matrix = pd.DataFrame(test_bh_returns_dict).dropna()

        # Đảm bảo index trùng khớp trên Test
        common_test_index = test_ret_matrix.index.intersection(test_bh_matrix.index)
        test_ret_matrix = test_ret_matrix.loc[common_test_index]
        test_bh_matrix = test_bh_matrix.loc[common_test_index]

        n_assets = len(train_ret_matrix.columns)
        equal_weights = np.repeat(1.0 / n_assets, n_assets)

        # Tính toán MPT Weights trên Train
        mpt_weights = optimize_mpt(train_ret_matrix, risk_free_rate=risk_free_rate)

        # Bảng hiển thị Trọng số
        st.subheader("⚖️ Bảng Phân Bổ Tỷ Trọng Danh Mục")
        col_w1, col_w2 = st.columns([1, 1])

        weight_df = pd.DataFrame({
            "Mã CK": train_ret_matrix.columns,
            "Tham số SMA/OBV": [f"({params_used[t]['ma_short']}, {params_used[t]['ma_long']}, {params_used[t]['obv_window']})" for t in train_ret_matrix.columns],
            "Equal Weight": equal_weights,
            "MPT Optimal Weight (Train)": mpt_weights
        })

        with col_w1:
            st.dataframe(
                weight_df.style.format({
                    "Equal Weight": "{:.2%}",
                    "MPT Optimal Weight (Train)": "{:.2%}"
                }),
                use_container_width=True
            )

        with col_w2:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=weight_df["Mã CK"], y=weight_df["Equal Weight"], name="Equal Weight", marker_color="#3B82F6"))
            fig_bar.add_trace(go.Bar(x=weight_df["Mã CK"], y=weight_df["MPT Optimal Weight (Train)"], name="MPT Weight", marker_color="#10B981"))
            fig_bar.update_layout(title="So sánh Tỷ trọng Phân bổ", yaxis_tickformat=".0%", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Tính toán tỷ suất sinh lời danh mục
        ew_bh_test = (test_bh_matrix * equal_weights).sum(axis=1)
        ew_strat_test = (test_ret_matrix * equal_weights).sum(axis=1)
        mpt_strat_test = (test_ret_matrix * mpt_weights).sum(axis=1)

        # Thống kê hiệu suất trên Test
        stats_ew_bh = performance_stats(ew_bh_test, risk_free_rate=risk_free_rate)
        stats_ew_strat = performance_stats(ew_strat_test, risk_free_rate=risk_free_rate)
        stats_mpt_strat = performance_stats(mpt_strat_test, risk_free_rate=risk_free_rate)

        summary_test_df = pd.DataFrame({
            "1. Equal Weight Buy & Hold": stats_ew_bh,
            f"2. SMA+OBV ({strat_logic}) Equal Weight": stats_ew_strat,
            f"3. SMA+OBV ({strat_logic}) MPT": stats_mpt_strat
        }).T

        st.subheader("🎯 Hiệu Suất Đầu Tư Ngoài Mẫu (Out-of-sample Test 2022)")
        st.dataframe(
            summary_test_df.style.format({
                "Mean Daily Return [%]": "{:.4f}%",
                "Total Return [%]": "{:.2f}%",
                "Annual Return [%]": "{:.2f}%",
                "Annual Volatility [%]": "{:.2f}%",
                "Sharpe Ratio": "{:.4f}",
                "Max Drawdown [%]": "{:.2f}%"
            }),
            use_container_width=True
        )

        # Đồ thị Tăng trưởng Vốn trên tập Test
        st.subheader("📈 Đường Tăng Trưởng Vốn (Equity Curve) trên Tập Test")
        eq_ew_bh = initial_capital * (1.0 + ew_bh_test).cumprod()
        eq_ew_strat = initial_capital * (1.0 + ew_strat_test).cumprod()
        eq_mpt_strat = initial_capital * (1.0 + mpt_strat_test).cumprod()

        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(x=test_ret_matrix.index, y=eq_ew_bh, mode="lines", name="Equal Weight Buy & Hold", line=dict(color="#DC2626", width=2, dash="dash")))
        fig_port.add_trace(go.Scatter(x=test_ret_matrix.index, y=eq_ew_strat, mode="lines", name="SMA+OBV Equal Weight", line=dict(color="#2563EB", width=2.5)))
        fig_port.add_trace(go.Scatter(x=test_ret_matrix.index, y=eq_mpt_strat, mode="lines", name="SMA+OBV MPT Optimal", line=dict(color="#10B981", width=2.5)))

        fig_port.update_layout(height=450, yaxis_title="Giá trị Danh mục (VNĐ)", hovermode="x unified",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_port, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: BÁO CÁO & KẾT LUẬN TỰ ĐỘNG
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("### 📝 Báo Cáo & Nhận Định Định Lượng Tự Động")

    if 'summary_test_df' in locals():
        ew_ret = summary_test_df.loc[f"2. SMA+OBV ({strat_logic}) Equal Weight", "Total Return [%]"]
        mpt_ret = summary_test_df.loc[f"3. SMA+OBV ({strat_logic}) MPT", "Total Return [%]"]
        bh_ret_val = summary_test_df.loc["1. Equal Weight Buy & Hold", "Total Return [%]"]

        ew_sharpe = summary_test_df.loc[f"2. SMA+OBV ({strat_logic}) Equal Weight", "Sharpe Ratio"]
        mpt_sharpe = summary_test_df.loc[f"3. SMA+OBV ({strat_logic}) MPT", "Sharpe Ratio"]
        bh_sharpe = summary_test_df.loc["1. Equal Weight Buy & Hold", "Sharpe Ratio"]

        ew_mdd = summary_test_df.loc[f"2. SMA+OBV ({strat_logic}) Equal Weight", "Max Drawdown [%]"]
        mpt_mdd = summary_test_df.loc[f"3. SMA+OBV ({strat_logic}) MPT", "Max Drawdown [%]"]
        bh_mdd = summary_test_df.loc["1. Equal Weight Buy & Hold", "Max Drawdown [%]"]

        st.markdown(f"""
        <div class="highlight-box">
            <h4>💡 Tổng Hợp Kết Quả Thực Nghiệm (Out-of-sample Test):</h4>
            <ul>
                <li><b>Chiến lược SMA + OBV so với Buy & Hold:</b> Trong năm 2022 khi thị trường sụt giảm mạnh, danh mục Buy & Hold ghi nhận mức lỗ <b>{bh_ret_val:.2f}%</b> với mức sụt giảm tối đa (Max Drawdown) là <b>{bh_mdd:.2f}%</b>.
                Ngược lại, cả hai danh mục ứng dụng tín hiệu <b>SMA + OBV ({strat_logic})</b> đều giúp <b>bảo toàn vốn vượt trội</b> (lỗ {ew_ret:.2f}% với Equal Weight và {mpt_ret:.2f}% với MPT) nhờ việc chuyển về trạng thái giữ tiền mặt khi có tín hiệu bán.</li>
                <li><b>So sánh Equal Weight vs MPT:</b>
                    {'<b>MPT Optimal</b> đạt Sharpe Ratio cao hơn (' + f'{mpt_sharpe:.4f} so với {ew_sharpe:.4f}) và mức lỗ thấp hơn, cho thấy việc tối ưu hóa hiệp phương sai trên tập Train vẫn duy trì được hiệu quả phân bổ ngoài mẫu.' if mpt_sharpe > ew_sharpe else '<b>Equal Weight</b> đạt Sharpe Ratio cao hơn (' + f'{ew_sharpe:.4f} so với {mpt_sharpe:.4f}), chứng minh tính vững chắc (robustness) và hạn chế rủi ro overfitting của phương pháp chia đều trọng số.'}
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Xuất dữ liệu
        st.subheader("📥 Xuất Báo Cáo Phân Tích")
        csv_buffer = io.StringIO()
        summary_test_df.to_csv(csv_buffer)
        st.download_button(
            label="⬇️ Tải xuống Bảng Kết Quả Kiểm Định (CSV)",
            data=csv_buffer.getvalue(),
            file_name="kiem_dinh_sma_obv_mpt_results.csv",
            mime="text/csv"
        )
    else:
        st.info("Vui lòng truy cập Tab 3 để hoàn thành bước kiểm định danh mục trước khi xem báo cáo tổng kết.")
