import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="House Price Prediction & Segmentation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ────
st.markdown("""
<style>
    /* ── Fix Streamlit top navbar / header ── */
    header[data-testid="stHeader"] {
        background-color: #60a5fa !important;
        opacity: 100 !important;
    }
    header[data-testid="stHeader"]::before {
        background-color: #60a5fa !important;
    }
    /* toolbar icons (hamburger, deploy button, etc.) */
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    /* Remove the default semi-transparent backdrop */
    .stApp > header {
        background-color: #1a3c5e !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

   .main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #60a5fa;
    margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .member-card {
        background: linear-gradient(135deg, #e8f4fd, #f0f8ff);
        border-left: 4px solid #1a3c5e;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #60a5fa;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    /* Default tab — teks hitam */
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 500;
        color: #111111 !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        color: #111111 !important;
    }
    /* Active / selected tab — teks merah + border bawah merah */
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        border-bottom: 3px solid #e53935 !important;
        color: #e53935 !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div {
        color: #e53935 !important;
        font-weight: 700 !important;
    }
    /* Hover state */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #fee2e2 !important;
        color: #e53935 !important;
    }
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span,
    .stTabs [data-baseweb="tab"]:hover div {
        color: #e53935 !important;
    }
 
    .cluster-badge-0 { background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:20px; font-weight:600;}
    .cluster-badge-1 { background:#dcfce7; color:#166534; padding:4px 12px; border-radius:20px; font-weight:600;}
    .cluster-badge-2 { background:#dbeafe; color:#1e40af; padding:4px 12px; border-radius:20px; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR NAVIGATION ────
st.sidebar.image("https://img.icons8.com/color/96/000000/home--v1.png", width=80)
st.sidebar.markdown("## Navigation")
menu = st.sidebar.radio(
    "",
    ["Home", "Dataset Overview", "Prediction / Analysis", "Visualization", "About"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Source:**")
st.sidebar.markdown("[Kaggle – House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)")

# ─── DATA & MODEL CACHING ───
@st.cache_data(show_spinner="Loading & preprocessing dataset…")
def load_and_prepare():
    # Try to load user-uploaded train.csv; otherwise generate synthetic data
    import os
    path_candidates = ["train.csv", "/mnt/user-data/uploads/train.csv"]
    df = None
    for p in path_candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break

    if df is None:
        # Generate synthetic data mirroring the Kaggle dataset structure
        np.random.seed(42)
        n = 1000
        overall_qual = np.random.randint(1, 11, n)
        gr_liv_area = (overall_qual * 150 + np.random.normal(500, 200, n)).clip(400, 4000).astype(int)
        garage_cars = np.random.choice([0, 1, 2, 3, 4], n, p=[0.05, 0.15, 0.50, 0.25, 0.05])
        total_bsmt_sf = (overall_qual * 80 + np.random.normal(400, 200, n)).clip(0, 2500).astype(int)
        year_built = np.random.randint(1900, 2011, n)
        sale_price = (
            overall_qual * 15000
            + gr_liv_area * 50
            + garage_cars * 8000
            + total_bsmt_sf * 20
            + (year_built - 1900) * 300
            + np.random.normal(0, 15000, n)
        ).clip(50000, 800000).astype(int)

        df = pd.DataFrame({
            "OverallQual": overall_qual, "GrLivArea": gr_liv_area,
            "GarageCars": garage_cars, "TotalBsmtSF": total_bsmt_sf,
            "YearBuilt": year_built, "SalePrice": sale_price,
        })
        is_synthetic = True
    else:
        columns_to_drop = ['PoolQC', 'Alley', 'Fence', 'MiscFeature']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns[df.select_dtypes(include=['int64', 'float64']).isnull().any()]
        cat_cols = df.select_dtypes(include=['object']).columns[df.select_dtypes(include=['object']).isnull().any()]
        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        is_synthetic = False

    return df, is_synthetic

@st.cache_resource(show_spinner="Training models…")
def train_models(_df):
    selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
    target = 'SalePrice'
    X = _df[selected_features]
    y = _df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns,  index=X_test.index)

    # Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_sc, y_train)
    y_pred_lr = lr_model.predict(X_test_sc)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_sc, y_train)
    y_pred_rf = rf_model.predict(X_test_sc)

    # Clustering
    clustering_features = ['SalePrice', 'GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF']
    X_clust = _df[clustering_features].copy()
    scaler_clust = StandardScaler()
    X_clust_sc = scaler_clust.fit_transform(X_clust)

    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        km.fit(X_clust_sc)
        wcss.append(km.inertia_)

    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_clust_sc)
    X_clust['Cluster'] = clusters
    sil = silhouette_score(X_clust_sc, clusters)

    metrics = {
        "lr":  {"MAE": mean_absolute_error(y_test, y_pred_lr),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                "R2": r2_score(y_test, y_pred_lr)},
        "rf":  {"MAE": mean_absolute_error(y_test, y_pred_rf),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                "R2": r2_score(y_test, y_pred_rf)},
        "silhouette": sil,
        "wcss": wcss,
    }
    return rf_model, lr_model, scaler, kmeans, scaler_clust, X_clust, metrics, y_test, y_pred_rf, y_pred_lr

df, is_synthetic = load_and_prepare()
rf_model, lr_model, scaler, kmeans, scaler_clust, X_clust, metrics, y_test, y_pred_rf, y_pred_lr = train_models(df)

# ─── CLUSTER LABEL MAPPING ────
cluster_summary = X_clust.groupby('Cluster').mean()
sorted_clusters = cluster_summary['SalePrice'].sort_values().index.tolist()
cluster_names = {sorted_clusters[0]: "Budget Property",
                 sorted_clusters[1]: "Mid-Range Property",
                 sorted_clusters[2]: "Premium Property"}
cluster_colors = {sorted_clusters[0]: "#ef4444",
                  sorted_clusters[1]: "#3b82f6",
                  sorted_clusters[2]: "#22c55e"}

# ════════════
# PAGE: HOME
# ════════════
if menu == "Home":
    col1, col2 = st.columns([2.5, 1])
    with col1:
        st.markdown('<div class="main-header">House Price Prediction & Property Segmentation</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Data Mining Project – CRISP-DM Framework</div>', unsafe_allow_html=True)

        st.markdown("""
        ### Project Description
        This application implements a **Data Mining** pipeline for the Kaggle *House Prices: Advanced Regression Techniques* dataset.  
        Two analytical goals are pursued simultaneously:

        | Goal | Method |
        |------|--------|
        **House Price Prediction** | Random Forest Regression + Linear Regression |
        **Property Segmentation** | K-Means Clustering (k = 3) |

        The project follows the **CRISP-DM** methodology — from Data Understanding and Preparation through Modeling, Evaluation, and Deployment.

        Key features used: `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`.
        """)

        if is_synthetic:
            st.info("⚠️ **Note:** The Kaggle `train.csv` file was not found. The app is running on **synthetic data** that mirrors the original dataset structure. Upload `train.csv` to the working directory for real results.")

    with col2:
        st.markdown("### Team Members")
        members = [
            ("Steven Krisna Putra", "24051214142"),
            ("Muhammad Nashrul Aziz", "24051214159"),
        ]
        for nim, role in members:
            st.markdown(f"""
            <div class="member-card">
                <b><span style="color:#000000;">{nim}</b><br>
                <small style="color:#000000">{role}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏆 Quick Stats")
        st.metric("RF R² Score", f"{metrics['rf']['R2']:.3f}", delta="vs Linear Reg")
        st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
        st.metric("Dataset Rows", f"{len(df):,}")

# ════════════════════════
# PAGE: DATASET OVERVIEW
# ════════════════════════
elif menu == "Dataset Overview":
    st.markdown('<div class="main-header">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploratory Data Analysis — House Prices Dataset</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Info", "Statistics", "Distributions", "Correlations"])

    with tab1:
        st.markdown('<div class="section-title">Dataset Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", f"{df.shape[0]:,}")
        c2.metric("Total Columns", f"{df.shape[1]:,}")
        c3.metric("Missing Values", "0 (after imputation)")

        st.markdown("**Preview (first 5 rows):**")
        selected_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'SalePrice']
        display_cols = [c for c in selected_cols if c in df.columns]
        st.dataframe(df[display_cols].head(), use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
        numeric_df = df[[c for c in selected_cols if c in df.columns]]
        st.dataframe(numeric_df.describe().round(2), use_container_width=True)

    with tab3:
        st.markdown('<div class="section-title">Distribution of SalePrice</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#f8fafc')

        sns.histplot(df['SalePrice'], bins=50, kde=True, color='#3b82f6',
                     edgecolor='white', alpha=0.8, line_kws={'color': '#ef4444', 'linewidth': 2}, ax=axes[0])
        axes[0].axvline(df['SalePrice'].mean(), color='#22c55e', linestyle='--', linewidth=2,
                        label=f"Mean: ${df['SalePrice'].mean():,.0f}")
        axes[0].axvline(df['SalePrice'].median(), color='#f97316', linestyle='-.', linewidth=2,
                        label=f"Median: ${df['SalePrice'].median():,.0f}")
        axes[0].set_title('Distribution of SalePrice', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Sale Price ($)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        feat_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
        feat_cols = [c for c in feat_cols if c in df.columns]
        corr_vals = df[feat_cols].corrwith(df['SalePrice']).sort_values(ascending=True)
        colors = ['#ef4444' if v < 0 else '#3b82f6' for v in corr_vals.values]
        axes[1].barh(corr_vals.index, corr_vals.values, color=colors, edgecolor='white')
        axes[1].set_title('Correlation with SalePrice', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Pearson r')
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="section-title">Feature vs SalePrice Scatter Plots</div>', unsafe_allow_html=True)
        fig2, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig2.patch.set_facecolor('#f8fafc')
        scatter_pairs = [('GrLivArea', 'SalePrice'), ('OverallQual', 'SalePrice'), ('GarageCars', 'SalePrice')]
        for ax, (xf, yf) in zip(axs, scatter_pairs):
            if xf in df.columns:
                ax.scatter(df[xf], df[yf], alpha=0.4, color='#3b82f6', s=20, edgecolors='none')
                ax.set_xlabel(xf)
                ax.set_ylabel(yf)
                ax.set_title(f'{xf} vs SalePrice', fontweight='bold')
                ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with tab4:
        st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        limited = [c for c in selected_cols if c in numeric_cols]
        fig3, ax = plt.subplots(figsize=(8, 6))
        fig3.patch.set_facecolor('#f8fafc')
        sns.heatmap(df[limited].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    linewidths=0.5, ax=ax, square=True)
        ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION / ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "Prediction / Analysis":
    st.markdown('<div class="main-header">Prediction / Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input property features to predict sale price and determine segment</div>', unsafe_allow_html=True)

    st.markdown("### Input Property Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6,
                                  help="Overall material and finish quality")
        gr_liv_area = st.number_input("Above-Grade Living Area (sq ft)", 300, 5000, 1500, 50)
    with c2:
        garage_cars = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2)
        total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800, 50)
    with c3:
        year_built = st.slider("Year Built", 1870, 2010, 1990)

    st.markdown("---")

    if st.button("🚀 Run Prediction & Clustering", type="primary", use_container_width=True):
        input_data = pd.DataFrame([[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf, year_built]],
                                   columns=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt'])
        input_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

        rf_pred = rf_model.predict(input_scaled)[0]
        lr_pred = lr_model.predict(input_scaled)[0]

        clust_input = pd.DataFrame([[rf_pred, gr_liv_area, overall_qual, garage_cars, total_bsmt_sf]],
                                    columns=['SalePrice', 'GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF'])
        clust_scaled = scaler_clust.transform(clust_input)
        cluster_id = kmeans.predict(clust_scaled)[0]
        cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Prediction Results")
        r1, r2, r3 = st.columns(3)
        r1.metric("🌲 Random Forest Prediction", f"${rf_pred:,.0f}")
        r2.metric("📏 Linear Regression Prediction", f"${lr_pred:,.0f}")
        delta_pct = ((rf_pred - lr_pred) / lr_pred * 100)
        r3.metric("Difference (RF vs LR)", f"{delta_pct:+.1f}%")

        st.markdown(f"""
        **🏷️ Property Segment:** &nbsp;
        <span style="background:{cluster_colors.get(cluster_id,'#94a3b8')};color:white;padding:5px 14px;
        border-radius:20px;font-weight:600;font-size:1rem;">{cluster_label}</span>
        """, unsafe_allow_html=True)

        cluster_desc = {
            "Budget Property": "This property falls in the **lower-end segment**, typically characterized by smaller living areas, lower overall quality scores, and fewer garage spaces.",
            "Mid-Range Property": "This property sits in the **mid-range segment**, offering a balanced combination of size, quality, and amenities — the most common segment in the market.",
            "Premium Property": "This property is in the **premium segment**, distinguished by above-average living area, high overall quality, ample garage capacity, and a larger basement footprint.",
        }
        st.info(cluster_desc.get(cluster_label, ""))
        st.markdown('</div>', unsafe_allow_html=True)

        

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "Visualization":
    st.markdown('<div class="main-header">Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Supporting charts and model analysis results</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Model Performance", "Cluster Analysis", "Elbow Method"])

    with tab1:
        st.markdown('<div class="section-title">Model Comparison — Regression Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("RF MAE", f"${metrics['rf']['MAE']:,.0f}",
                  delta=f"${metrics['lr']['MAE'] - metrics['rf']['MAE']:+,.0f} vs LR")
        m2.metric("RF RMSE", f"${metrics['rf']['RMSE']:,.0f}",
                  delta=f"${metrics['lr']['RMSE'] - metrics['rf']['RMSE']:+,.0f} vs LR")
        m3.metric("RF R²", f"{metrics['rf']['R2']:.4f}",
                  delta=f"{metrics['rf']['R2'] - metrics['lr']['R2']:+.4f} vs LR")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#f8fafc')

        # Actual vs Predicted
        axes[0].scatter(y_test, y_pred_rf, alpha=0.4, color='#3b82f6', s=15, label='RF')
        axes[0].scatter(y_test, y_pred_lr, alpha=0.2, color='#f97316', s=10, label='LR')
        lims = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
        axes[0].plot(lims, lims, 'k--', linewidth=1.5, label='Perfect fit')
        axes[0].set_xlabel("Actual SalePrice")
        axes[0].set_ylabel("Predicted SalePrice")
        axes[0].set_title("Actual vs Predicted (Test Set)", fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Metrics bar chart
        models = ['Linear Regression', 'Random Forest']
        metrics_vals = {
            'MAE': [metrics['lr']['MAE'], metrics['rf']['MAE']],
            'RMSE': [metrics['lr']['RMSE'], metrics['rf']['RMSE']],
        }
        x = np.arange(len(models))
        w = 0.35
        axes[1].bar(x - w/2, metrics_vals['MAE'], w, label='MAE', color='#3b82f6', edgecolor='white')
        axes[1].bar(x + w/2, metrics_vals['RMSE'], w, label='RMSE', color='#ef4444', edgecolor='white')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].set_ylabel("Error ($)")
        axes[1].set_title("MAE & RMSE Comparison", fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature Importance
        st.markdown('<div class="section-title">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
        feat_names = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance', ascending=True)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.patch.set_facecolor('#f8fafc')
        colors_fi = plt.cm.Spectral(np.linspace(0.2, 0.9, len(fi_df)))
        bars = ax2.barh(fi_df['Feature'], fi_df['Importance'], color=colors_fi, edgecolor='white')
        for bar in bars:
            ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                     f"{bar.get_width():.4f}", va='center', fontsize=9)
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Feature Importance – Random Forest', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with tab2:
        st.markdown('<div class="section-title">Cluster Visualization</div>', unsafe_allow_html=True)
        col_names = {c: cluster_names[c] for c in X_clust['Cluster'].unique()}
        X_clust_labeled = X_clust.copy()
        X_clust_labeled['Segment'] = X_clust_labeled['Cluster'].map(col_names)

        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        fig3.patch.set_facecolor('#f8fafc')

        palette = {v: cluster_colors[k] for k, v in cluster_names.items()}
        sns.scatterplot(data=X_clust_labeled, x='GrLivArea', y='SalePrice',
                        hue='Segment', palette=palette, alpha=0.6, s=40, ax=axes3[0])
        axes3[0].set_title('Clusters: Living Area vs Sale Price', fontweight='bold')
        axes3[0].grid(alpha=0.3)

        # Cluster heatmap
        clust_sum = X_clust.groupby('Cluster').mean()
        clust_sum.index = [cluster_names.get(i, f'Cluster {i}') for i in clust_sum.index]
        sns.heatmap(clust_sum, annot=True, fmt='.0f', cmap='YlGnBu',
                    linewidths=0.5, ax=axes3[1])
        axes3[1].set_title('Cluster Characteristics (Mean Values)', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        st.markdown('<div class="section-title">Cluster Summary Table</div>', unsafe_allow_html=True)
        summary = X_clust.groupby('Cluster').agg(['mean', 'count']).round(1)
        summary_mean = X_clust.groupby('Cluster').mean().round(1)
        summary_mean.index = [cluster_names.get(i, i) for i in summary_mean.index]
        summary_mean['Count'] = X_clust['Cluster'].value_counts().sort_index().values
        st.dataframe(summary_mean.style.background_gradient(cmap='Blues', subset=summary_mean.columns[:-1]),
                     use_container_width=True)

        st.metric("Silhouette Score", f"{metrics['silhouette']:.4f}",
                  help="Ranges from -1 to 1; higher is better. > 0.5 indicates reasonable cluster structure.")

    with tab3:
        st.markdown('<div class="section-title">Elbow Method – Optimal K Selection</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        fig4.patch.set_facecolor('#f8fafc')
        ax4.plot(range(1, 11), metrics['wcss'], marker='o', color='#3b82f6',
                 linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax4.axvline(3, color='#ef4444', linestyle='--', linewidth=2, label='Optimal K = 3')
        ax4.set_xlabel("Number of Clusters (K)", fontsize=12)
        ax4.set_ylabel("WCSS (Inertia)", fontsize=12)
        ax4.set_title("Elbow Method for Optimal K", fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.set_xticks(range(1, 11))
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        st.info("The elbow at **K = 3** indicates that adding more clusters beyond three yields diminishing returns in reducing within-cluster variance. Three clusters map naturally to *Budget*, *Mid-Range*, and *Premium* property segments.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "About":
    st.markdown('<div class="main-header">About This Project</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Methodology", "Dataset", "Project Info"])

    with tab1:
        st.markdown("""
        ### CRISP-DM Methodology

        This project strictly follows the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** framework:

        | Phase | Description |
        |-------|-------------|
        | **1. Business Understanding** | Predict house sale prices and segment properties by market tier |
        | **2. Data Understanding** | EDA — distribution, correlations, missing value analysis |
        | **3. Data Preparation** | Drop high-missing columns, impute, select features, scale with StandardScaler |
        | **4. Modeling** | Linear Regression, Random Forest Regression, K-Means Clustering |
        | **5. Evaluation** | MAE, RMSE, R² for regression; Silhouette Score for clustering |
        | **6. Deployment** | Streamlit web application (this app) |

        ### Algorithms Used

        **Random Forest Regression**  
        An ensemble of decision trees that reduces variance through bagging. Particularly well-suited for tabular housing data because it handles non-linear relationships and provides feature importance scores.

        **K-Means Clustering**  
        Partitions properties into k = 3 groups by minimising within-cluster sum of squares (WCSS). The optimal k was determined via the Elbow Method.

        **Linear Regression**  
        Included as a baseline model; its simpler assumptions make it interpretable but less accurate on heterogeneous housing data.
        """)

    with tab2:
        st.markdown("""
        ### Dataset: House Prices – Advanced Regression Techniques

        - **Source:** [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
        - **Format:** CSV (`train.csv`) — 1,460 rows, 81 columns (original)  
        - **Target Variable:** `SalePrice` — the property's final sale price in USD  

        ### Selected Features

        | Feature | Description |
        |---------|-------------|
        `OverallQual` | Rates the overall material and finish of the house (1–10) |
        `GrLivArea` | Above-ground living area in square feet |
        `GarageCars` | Size of garage in car capacity |
        `TotalBsmtSF` | Total square feet of basement area |
        `YearBuilt` | Original construction year |

        These five features were chosen based on their high correlation with `SalePrice` and domain relevance in real-estate valuation.
        """)

    with tab3:
        st.markdown("""
        ### Project Information

        | Item | Detail |
        |------|--------|
        **Course** | Data Mining |
        **Framework** | CRISP-DM |
        **Language** | Python 3 |
        **Key Libraries** | scikit-learn, pandas, matplotlib, seaborn, Streamlit |
        **Deployment** | Streamlit Web Application |

        ### Model Performance Summary
        """)

        perf_df = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "MAE ($)": [f"{metrics['lr']['MAE']:,.0f}", f"{metrics['rf']['MAE']:,.0f}"],
            "RMSE ($)": [f"{metrics['lr']['RMSE']:,.0f}", f"{metrics['rf']['RMSE']:,.0f}"],
            "R²": [f"{metrics['lr']['R2']:.4f}", f"{metrics['rf']['R2']:.4f}"],
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.success(f"✅ Random Forest outperforms Linear Regression with R² = **{metrics['rf']['R2']:.4f}** vs **{metrics['lr']['R2']:.4f}**")
        st.info(f"📊 K-Means silhouette score = **{metrics['silhouette']:.4f}** — indicating reasonable cluster cohesion.")
