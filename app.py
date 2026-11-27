from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cleaned_employee_salary.csv"
MODEL_PATH = ROOT / "models" / "linear_regression_model.joblib"
MODEL_WEIGHTS_PATH = ROOT / "models" / "model_weights.json"
PREPROCESSING_PATH = ROOT / "models" / "inference_preprocessing.json"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
COMPARISON_PATH = ROOT / "models" / "evaluation_comparison.csv"
REQUIRED_COLUMNS = [
    "Department", "Department_Name", "Division", "Gender", "Base_Salary",
    "Overtime_Pay", "Longevity_Pay", "Grade",
]
FEATURES = [c for c in REQUIRED_COLUMNS if c != "Base_Salary"]
COLORS = ["#8b7bff", "#41d9f2", "#30d6b0", "#ffb86b", "#f47bba", "#6399ff"]

st.set_page_config(
    page_title="Employee Salary Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root { --bg:#080d19; --surface:#101a2b; --surface-2:#142039; --line:rgba(155,177,219,.14); --muted:#91a1bb; --text:#f2f5fc; --blue:#56a8ff; --violet:#9b8cff; --cyan:#48d7ed; --teal:#4bd6b2; }
        html,body,[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 82% -12%,rgba(101,78,224,.18),transparent 34%),radial-gradient(ellipse at 4% 20%,rgba(37,147,184,.10),transparent 30%),var(--bg);color:var(--text); }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stAppViewContainer"] .main .block-container { max-width:1580px;padding:1.65rem clamp(1rem,3vw,2.8rem) 2.6rem; }
        * { font-family:'DM Sans',sans-serif; }
        h1,h2,h3,h4 { font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:var(--text); }
        h1 { font-size:clamp(1.8rem,3vw,2.45rem)!important;font-weight:800!important; }
        h2 { font-size:1.45rem!important;font-weight:700!important; }
        h3 { font-size:1.05rem!important;font-weight:700!important; }
        p,li,label { color:#bdc8da; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#0e1728,#0a1120 85%);border-right:1px solid var(--line); }
        [data-testid="stSidebar"]>div { padding-top:1rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:10px;padding:.38rem .55rem;transition:background .18s ease; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(115,126,255,.11); }
        [data-testid="stSidebar"] [role="radio"][aria-checked="true"] { background:linear-gradient(90deg,rgba(109,101,232,.2),rgba(67,173,208,.08));border-radius:10px; }
        .hero { position:relative;overflow:hidden;padding:clamp(1.4rem,3vw,2.2rem);margin:.15rem 0 1.35rem;border-radius:22px;border:1px solid rgba(147,152,255,.2);background:linear-gradient(115deg,rgba(42,49,94,.75),rgba(15,31,52,.88) 57%,rgba(11,47,62,.72));box-shadow:0 18px 52px rgba(0,0,0,.2); }
        .hero:after { content:'';position:absolute;width:270px;height:270px;right:-90px;top:-170px;border-radius:50%;background:radial-gradient(circle,rgba(72,215,237,.22),transparent 68%);pointer-events:none; }
        .eyebrow { text-transform:uppercase;letter-spacing:.15em;font-size:.68rem;color:#a99bff;font-weight:700;margin-bottom:.55rem; }
        .hero-title { font-family:'Manrope',sans-serif;font-size:clamp(1.65rem,3.2vw,2.35rem);font-weight:800;letter-spacing:-.045em;color:#f6f7ff;line-height:1.16; }
        .hero-subtitle { color:#bac6d9;margin-top:.52rem;font-size:.94rem; }
        .hero-chip { display:inline-flex;align-items:center;gap:.42rem;margin-top:1rem;padding:.34rem .64rem;color:#baf1e2;background:rgba(62,211,175,.08);border:1px solid rgba(62,211,175,.19);border-radius:999px;font-size:.71rem; }
        .section-kicker { color:#8d9db9;text-transform:uppercase;letter-spacing:.14em;font-size:.66rem;font-weight:700;margin-bottom:.28rem; }
        .panel { background:linear-gradient(145deg,rgba(19,30,50,.94),rgba(13,22,38,.96));border:1px solid var(--line);border-radius:17px;padding:1.05rem 1.15rem;box-shadow:0 10px 28px rgba(0,0,0,.12); }
        .callout { border:1px solid rgba(155,140,255,.16);border-left:3px solid var(--violet);padding:.8rem 1rem;border-radius:0 12px 12px 0;background:linear-gradient(90deg,rgba(139,123,255,.09),rgba(72,215,237,.035));color:#c7d1e2;font-size:.88rem; }
        .tag { padding:.24rem .52rem;border-radius:999px;border:1px solid var(--line);color:#bdc8da;font-size:.72rem; }
        .kpi-grid { display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:.45rem 0 .72rem; }
        .kpi-card { min-width:0;min-height:112px;position:relative;overflow:hidden;padding:1rem 1.05rem;border:1px solid rgba(151,175,221,.14);border-radius:16px;background:linear-gradient(145deg,rgba(20,32,54,.97),rgba(12,21,38,.97));box-shadow:0 10px 26px rgba(0,0,0,.16); }
        .kpi-card:before { content:'';position:absolute;inset:0 auto 0 0;width:3px;background:linear-gradient(180deg,var(--violet),var(--cyan));opacity:.88; }
        .kpi-label { color:#9cacc5;font-size:.75rem;font-weight:600;letter-spacing:.025em;margin-bottom:.65rem; }
        .kpi-value { display:block;color:#f4f7ff;font-family:'Manrope',sans-serif;font-size:clamp(1.18rem,1.7vw,1.62rem);font-weight:800;line-height:1.2;letter-spacing:-.035em;white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere;font-variant-numeric:tabular-nums; }
        .kpi-detail { color:#7f90ad;font-size:.7rem;margin-top:.45rem; }
        div[data-testid="stDataFrame"] { border:1px solid var(--line);border-radius:13px;overflow:hidden; }
        div[data-testid="stPlotlyChart"] { background:linear-gradient(145deg,rgba(15,25,43,.84),rgba(12,20,36,.78));border:1px solid var(--line);border-radius:16px;padding:.35rem; }
        .stButton button,.stDownloadButton button { border-radius:10px;border:1px solid rgba(139,123,255,.34);background:linear-gradient(110deg,#7168e8,#398fc5);color:white;font-weight:700;padding:.55rem .95rem; }
        .stButton button:hover,.stDownloadButton button:hover { border-color:var(--cyan);color:white; }
        [data-testid="stSelectbox"] div[data-baseweb="select"],[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input { border-radius:9px; }
        [data-testid="stCaptionContainer"] { color:var(--muted); }
        hr { border-color:var(--line); }
        footer { visibility:hidden; }
        @media(max-width:1000px) { .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr));gap:10px; } }
        @media(max-width:600px) { [data-testid="stAppViewContainer"] .main .block-container { padding:.9rem .8rem 2rem; } .kpi-grid { grid-template-columns:1fr;gap:8px; } .kpi-card { min-height:92px; } .hero { border-radius:17px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_metrics() -> dict:
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"model": "Linear Regression", "target": "Base_Salary", "mae": None, "r2": None, "source": "Unavailable"}


def load_dataset(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            source = "Uploaded CSV"
        elif DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            source = DATA_PATH.name
        else:
            return None, None
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
        missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_columns:
            raise ValueError("Missing required columns: " + ", ".join(missing_columns))
        for col in ["Base_Salary", "Overtime_Pay", "Longevity_Pay"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Base_Salary"]).copy()
        return df, source
    except Exception as exc:
        st.sidebar.error(f"Could not read dataset: {exc}")
        return None, None


@st.cache_resource(show_spinner="Loading the saved model preprocessing…")
def get_inference_assets():
    weights = json.loads(MODEL_WEIGHTS_PATH.read_text(encoding="utf-8"))
    preprocessing = json.loads(PREPROCESSING_PATH.read_text(encoding="utf-8"))
    mean = np.asarray(preprocessing["mean"], dtype=float)
    scale = np.asarray(preprocessing["scale"], dtype=float)
    columns = preprocessing["feature_columns"]
    bounds = preprocessing["clip_bounds"]
    categorical = preprocessing["categorical_features"]
    if len(mean) != len(columns) or len(scale) != len(columns):
        raise ValueError("Saved scaler statistics do not match the saved model feature schema.")
    coefficients = np.asarray(weights["coefficients"], dtype=float)
    if len(coefficients) != len(columns):
        raise ValueError("Saved regression coefficients do not match the notebook feature schema.")
    estimator = None
    estimator_source = "verified coefficient export"
    try:
        import joblib

        if MODEL_PATH.exists():
            candidate = joblib.load(MODEL_PATH)
            if getattr(candidate, "n_features_in_", len(columns)) != len(columns):
                raise ValueError("The serialized estimator feature count does not match the saved scaler schema.")
            estimator = candidate
            estimator_source = "serialized Linear Regression estimator"
    except ImportError:
        # The supplied coefficient export is an exact representation of this estimator.
        pass
    return estimator, estimator_source, coefficients, float(weights["intercept"]), mean, scale, columns, bounds, categorical


def predict_salary(values: dict, df: pd.DataFrame) -> float:
    estimator, _, coefficients, intercept, mean, scale, columns, bounds, categorical = get_inference_assets()
    row = pd.DataFrame([values], columns=FEATURES)
    for col, (lower, upper) in bounds.items():
        row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0).clip(lower, upper)
    encoded = pd.get_dummies(row, columns=categorical, dtype=int).reindex(columns=columns, fill_value=0)
    scaled = (encoded.to_numpy(dtype=float) - mean) / scale
    if estimator is not None:
        return float(estimator.predict(scaled)[0])
    return float(np.asarray(scaled @ coefficients + intercept).item())


def money(value) -> str:
    return "—" if pd.isna(value) else f"${value:,.0f}"


def friendly_gender(value) -> str:
    normalized = str(value).strip().casefold()
    if normalized in {"f", "female"}:
        return "Female"
    if normalized in {"m", "male"}:
        return "Male"
    return str(value).strip()


def gender_filter_options(series: pd.Series) -> dict[str, list[str]]:
    """Map readable gender labels to the exact categories stored in the CSV."""
    options: dict[str, list[str]] = {}
    for raw in series.dropna().astype(str).unique():
        label = friendly_gender(raw)
        options.setdefault(label, []).append(raw)
    return dict(sorted(options.items()))


def render_kpis(items: list[tuple[str, str, str]]) -> None:
    """Render full-width, wrapping KPI cards instead of narrow Streamlit metrics."""
    cards = "".join(
        "<div class='kpi-card'>"
        f"<div class='kpi-label'>{escape(label)}</div>"
        f"<div class='kpi-value'>{escape(value)}</div>"
        f"<div class='kpi-detail'>{escape(detail)}</div>"
        "</div>"
        for label, value, detail in items
    )
    st.markdown(f"<div class='kpi-grid'>{cards}</div>", unsafe_allow_html=True)


def plot_layout(fig, height=360):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans", "color": "#b5c1d7", "size": 11},
        margin={"l": 12, "r": 12, "t": 25, "b": 12},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#b5c1d7"}},
        xaxis={"gridcolor": "rgba(152,174,222,.09)", "zerolinecolor": "rgba(152,174,222,.12)"},
        yaxis={"gridcolor": "rgba(152,174,222,.09)", "zerolinecolor": "rgba(152,174,222,.12)"},
    )
    return fig


def header() -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">People analytics · model intelligence</div>'
        '<div class="hero-title">Employee Salary Analysis Dashboard</div>'
        '<div class="hero-subtitle">Machine Learning&nbsp; | &nbsp;Linear Regression&nbsp; | &nbsp;Employee Salary Analytics</div>'
        '<div class="hero-chip">● &nbsp; Salary insights, grounded in your training project</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, kicker: str | None = None) -> None:
    if kicker:
        st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(title)


def overview(df: pd.DataFrame | None, metrics: dict, source: str | None, comparison: pd.DataFrame):
    header()
    if df is None:
        st.info("The cleaned notebook export is missing. Generate `data/cleaned_employee_salary.csv` to enable salary analysis. The notebook’s verified model results are shown below.")
        render_kpis([
            ("Cleaned records", "Awaiting export", "Notebook-cleaned dataset"),
            ("Model", metrics.get("model", "Linear Regression"), "Existing trained model"),
            ("MAE", f"${metrics['mae']:,.2f}" if metrics.get("mae") is not None else "—", "Held-out test set"),
            ("R² score", f"{metrics['r2']:.6f}" if metrics.get("r2") is not None else "—", "Held-out test set"),
        ])
    else:
        salary = df["Base_Salary"].dropna()
        render_kpis([
            ("Total employees", f"{len(df):,}", "Cleaned employee records"),
            ("Average salary", money(salary.mean()), "Mean base salary"),
            ("Median salary", money(salary.median()), "50th percentile"),
            ("Highest salary", money(salary.max()), "Maximum base salary"),
        ])
        render_kpis([
            ("Lowest salary", money(salary.min()), "Minimum base salary"),
            ("MAE", f"${metrics['mae']:,.2f}" if metrics.get("mae") is not None else "—", "Average absolute error"),
            ("R² score", f"{metrics['r2']:.6f}" if metrics.get("r2") is not None else "—", "Held-out test fit"),
        ])
    st.markdown("")
    left, right = st.columns([1.45, 1])
    with left:
        section("Salary at a glance", "Overview")
        if df is not None:
            fig = px.histogram(df, x="Base_Salary", nbins=38, color_discrete_sequence=[COLORS[0]], opacity=.88)
            fig.update_traces(marker_line_width=0, hovertemplate="Salary: $%{x:,.0f}<br>Employees: %{y}<extra></extra>")
            fig.update_xaxes(title="Base salary", tickprefix="$", tickformat=",.0f")
            fig.update_yaxes(title="Employees")
            st.plotly_chart(plot_layout(fig, 310), width="stretch")
        else:
            fig = px.scatter(comparison, x="Actual Salary", y="Predicted Salary", color_discrete_sequence=[COLORS[1]], opacity=.75)
            fig.update_xaxes(title="Actual salary", tickprefix="$", tickformat=",.0f")
            fig.update_yaxes(title="Predicted salary", tickprefix="$", tickformat=",.0f")
            st.plotly_chart(plot_layout(fig, 310), width="stretch")
    with right:
        section("Model snapshot", "Performance")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Linear Regression**  ·  target `Base_Salary`")
        st.markdown(f"<div style='font-size:2rem;font-weight:800;color:#46d8ef'>{metrics.get('r2', 0):.3f} <span style='font-size:.85rem;color:#94a3be'>R² score</span></div>", unsafe_allow_html=True)
        st.markdown(f"**{money(metrics.get('mae'))}** average absolute error on the held-out test set.")
        st.caption("MAE is the average size of the prediction error in dollars. R² indicates how much of the salary variation the model explains.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")
        if df is not None:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(f"**{df['Department'].nunique():,}** departments &nbsp; · &nbsp; **{df['Division'].nunique():,}** divisions")
            st.caption(f"Source file: {source} · {len(df.columns)} columns")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("Evaluation values are recorded in the completed project notebook. No metrics are estimated here.")
    if df is not None:
        section("Explore the workforce", "Quick look")
        c1, c2 = st.columns(2)
        dept = df.groupby("Department", as_index=False).agg(Employees=("Base_Salary", "size"), Average_Salary=("Base_Salary", "mean")).sort_values("Employees", ascending=False).head(8)
        fig = px.bar(dept.sort_values("Average_Salary"), x="Average_Salary", y="Department", orientation="h", color="Average_Salary", color_continuous_scale=[[0, "#5751a8"], [1, "#43d7e8"]], hover_data={"Employees": True, "Average_Salary": ":$,.0f"})
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title="Average base salary", tickprefix="$", tickformat=",.0f")
        fig.update_yaxes(title="")
        c1.plotly_chart(plot_layout(fig, 300), width="stretch")
        gender = df.assign(Gender_Display=df["Gender"].map(friendly_gender)).groupby("Gender_Display", as_index=False)["Base_Salary"].mean()
        fig = px.bar(gender, x="Gender_Display", y="Base_Salary", color="Gender_Display", color_discrete_sequence=COLORS, text_auto="$,.0f")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="Average base salary", tickprefix="$", tickformat=",.0f")
        c2.plotly_chart(plot_layout(fig, 300), width="stretch")
        st.markdown("")
        section("Business observations", "From cleaned records")
        department_summary = df.groupby("Department_Name").agg(Average_Salary=("Base_Salary", "mean"), Employees=("Base_Salary", "size")).query("Employees >= 10")
        gender_summary = df.groupby("Gender")["Base_Salary"].mean().sort_values(ascending=False)
        overtime_share = (df["Overtime_Pay"].fillna(0) > 0).mean() * 100
        i1, i2, i3 = st.columns(3)
        if not department_summary.empty:
            top_department = department_summary["Average_Salary"].idxmax()
            top_value = department_summary.loc[top_department, "Average_Salary"]
            i1.markdown(f'<div class="panel"><div class="section-kicker">Highest department mean</div><b>{top_department}</b><br><span style="color:#46d8ef">{money(top_value)}</span><br><small>At least 10 employees in the cleaned data</small></div>', unsafe_allow_html=True)
        if len(gender_summary) >= 2:
            gap = gender_summary.iloc[0] - gender_summary.iloc[-1]
            i2.markdown(f'<div class="panel"><div class="section-kicker">Gender mean difference</div><b>{money(gap)}</b><br><small>Unadjusted descriptive difference; not a like-for-like comparison</small></div>', unsafe_allow_html=True)
        i3.markdown(f'<div class="panel"><div class="section-kicker">Employees with overtime</div><b>{overtime_share:.1f}%</b><br><small>Share with overtime pay greater than zero</small></div>', unsafe_allow_html=True)


def salary_analysis(df: pd.DataFrame | None):
    section("Salary analysis", "Compensation patterns")
    if df is None:
        st.info("The cleaned notebook export is unavailable. Add `data/cleaned_employee_salary.csv` or upload the cleaned CSV in the sidebar.")
        return
    departments = sorted(df["Department"].dropna().astype(str).unique())
    filter_col1, filter_col2 = st.columns([1.4, 1])
    selected = filter_col1.multiselect("Departments", departments, default=[])
    gender_options = gender_filter_options(df["Gender"])
    selected_gender = filter_col2.multiselect("Gender", list(gender_options), default=[])
    if "CCL" in departments:
        ccl_names = df.loc[df["Department"].astype(str).eq("CCL"), "Department_Name"].dropna().astype(str).unique()
        if len(ccl_names):
            st.caption(f"Department codes come from the `Department` column. CCL is {ccl_names[0]}. Female/Male labels map to source values F/M.")
    view = df.copy()
    if selected:
        view = view[view["Department"].astype(str).isin(selected)]
    if selected_gender:
        selected_raw_genders = [raw for label in selected_gender for raw in gender_options[label]]
        view = view[view["Gender"].astype(str).isin(selected_raw_genders)]
    if view.empty:
        st.warning("No matching employees found for the selected department and gender.")
        return
    st.caption(f"Showing {len(view):,} cleaned employee records for the selected filters.")
    c1, c2 = st.columns(2)
    fig = px.histogram(view, x="Base_Salary", nbins=42, marginal="box", color_discrete_sequence=[COLORS[0]], opacity=.82)
    fig.update_xaxes(title="Base salary", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Employees")
    c1.plotly_chart(plot_layout(fig), width="stretch")
    dept = view.groupby("Department", as_index=False).agg(Average_Salary=("Base_Salary", "mean"), Employees=("Base_Salary", "size")).sort_values("Average_Salary", ascending=False).head(18)
    fig = px.bar(dept.sort_values("Average_Salary"), x="Average_Salary", y="Department", orientation="h", color="Average_Salary", color_continuous_scale=[[0, "#5552a7"], [1, "#42d9ee"]], hover_data={"Employees": True, "Average_Salary": ":$,.0f"})
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title="Average base salary", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="")
    c2.plotly_chart(plot_layout(fig, 440), width="stretch")
    c3, c4 = st.columns(2)
    gender_view = view.assign(Gender_Display=view["Gender"].map(friendly_gender))
    gender = gender_view.groupby("Gender_Display", as_index=False).agg(Average_Salary=("Base_Salary", "mean"), Median_Salary=("Base_Salary", "median"), Employees=("Base_Salary", "size"))
    fig = px.bar(gender, x="Gender_Display", y="Average_Salary", color="Gender_Display", color_discrete_sequence=COLORS, text_auto="$,.0f", hover_data={"Median_Salary": ":$,.0f", "Employees": True})
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Average base salary", tickprefix="$", tickformat=",.0f")
    c3.plotly_chart(plot_layout(fig), width="stretch")
    division = view.groupby("Division", as_index=False).agg(Average_Salary=("Base_Salary", "mean"), Employees=("Base_Salary", "size")).query("Employees >= 5").nlargest(14, "Average_Salary")
    fig = px.bar(division.sort_values("Average_Salary"), x="Average_Salary", y="Division", orientation="h", color="Average_Salary", color_continuous_scale=[[0, "#396b91"], [1, "#3ad5b0"]], hover_data={"Employees": True})
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title="Average base salary", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="")
    c4.plotly_chart(plot_layout(fig, 420), width="stretch")
    c5, c6 = st.columns(2)
    grade = view.dropna(subset=["Grade"]).groupby("Grade", as_index=False).agg(Average_Salary=("Base_Salary", "mean"), Employees=("Base_Salary", "size")).query("Employees >= 5").nlargest(18, "Average_Salary")
    fig = px.bar(grade.sort_values("Average_Salary"), x="Average_Salary", y="Grade", orientation="h", color="Average_Salary", color_continuous_scale=[[0, "#5552a7"], [1, "#46d8ef"]], hover_data={"Employees": True, "Average_Salary": ":$,.0f"})
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title="Average base salary", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Grade classification · job-level proxy")
    c5.plotly_chart(plot_layout(fig, 380), width="stretch")
    overtime = view.assign(Gender_Display=view["Gender"].map(friendly_gender)).groupby("Gender_Display", as_index=False).agg(Overtime_Pay=("Overtime_Pay", "mean"), Longevity_Pay=("Longevity_Pay", "mean"))
    fig = go.Figure()
    fig.add_bar(x=overtime["Gender_Display"], y=overtime["Overtime_Pay"], name="Average overtime", marker_color=COLORS[1])
    fig.add_bar(x=overtime["Gender_Display"], y=overtime["Longevity_Pay"], name="Average longevity", marker_color=COLORS[2])
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Average pay", tickprefix="$", tickformat=",.0f")
    c6.plotly_chart(plot_layout(fig, 380), width="stretch")
    st.caption("Division and grade comparisons require at least five employees per group. The source has Grade but no job title, experience, or education fields; Grade is shown as a classification proxy, not a job title.")


def model_performance(metrics: dict, comparison: pd.DataFrame):
    section("Model performance", "Held-out test results")
    render_kpis([
        ("Model", metrics.get("model", "Linear Regression"), "Existing trained estimator"),
        ("Target", "Base Salary", "Regression target"),
        ("Mean Absolute Error", f"${metrics['mae']:,.2f}" if metrics.get("mae") is not None else "—", "Average absolute error · USD"),
        ("R² score", f"{metrics['r2']:.6f}" if metrics.get("r2") is not None else "—", "Held-out test set"),
    ])
    st.markdown('<div class="callout"><b>Reading the metrics</b><br>MAE is the average distance between a prediction and the actual salary, measured in dollars. R² compares the model with a baseline that predicts the test-set average; higher values indicate that the model explains more salary variation. These are the evaluation results saved from your notebook.</div>', unsafe_allow_html=True)
    st.markdown("")
    section("Actual vs predicted salary", "Prediction quality")
    if comparison.empty:
        st.info("No actual/predicted test rows were found in the project artifacts.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comparison["Actual Salary"], y=comparison["Predicted Salary"], mode="markers", name="Test employees", marker={"color": COLORS[1], "opacity": .63, "size": 9, "line": {"color": "rgba(255,255,255,.25)", "width": .5}}, customdata=np.arange(1, len(comparison) + 1), hovertemplate="Test row %{customdata}<br>Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>"))
    extent = [min(comparison["Actual Salary"].min(), comparison["Predicted Salary"].min()), max(comparison["Actual Salary"].max(), comparison["Predicted Salary"].max())]
    fig.add_trace(go.Scatter(x=extent, y=extent, mode="lines", name="Perfect prediction", line={"color": COLORS[0], "dash": "dash", "width": 2}, hoverinfo="skip"))
    fig.update_xaxes(title="Actual salary", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Predicted salary", tickprefix="$", tickformat=",.0f", scaleanchor="x", scaleratio=1)
    st.plotly_chart(plot_layout(fig, 480), width="stretch")
    st.caption("This table contains the 10 actual test examples preserved in the notebook output. The MAE and R² above were evaluated on the full held-out test set.")
    st.dataframe(comparison.style.format({"Actual Salary": "${:,.2f}", "Predicted Salary": "${:,.2f}"}), width="stretch", hide_index=True)


def prediction_page(df: pd.DataFrame | None):
    section("Salary prediction", "Saved Linear Regression estimator")
    if df is None:
        st.warning("The notebook’s fitted preprocessing artifact is missing. Run its final export cell, then restart the app. The saved estimator is never retrained.")
        return
    if not MODEL_WEIGHTS_PATH.exists():
        st.error("The supplied Linear Regression coefficient artifact is missing from `models/model_weights.json`.")
        return
    if not PREPROCESSING_PATH.exists():
        st.warning("The saved scaler and feature schema are missing. Run `python clean_data.py` to create `models/inference_preprocessing.json`.")
        return
    st.markdown('<div class="callout">Enter employee attributes to estimate base salary with the saved Linear Regression model. Categorical levels outside its training vocabulary are encoded as all-zero categories, matching the original train/test feature alignment.</div>', unsafe_allow_html=True)
    departments = sorted(df["Department"].dropna().astype(str).unique())
    dept = st.selectbox("Department", departments, key="pred_department")
    department_names = sorted(df.loc[df["Department"].astype(str) == dept, "Department_Name"].dropna().astype(str).unique())
    department_name = st.selectbox("Department name", department_names or sorted(df["Department_Name"].dropna().astype(str).unique()), key="pred_department_name")
    division = st.selectbox("Division", sorted(df["Division"].dropna().astype(str).unique()), key="pred_division")
    gender = st.selectbox("Gender", sorted(df["Gender"].dropna().astype(str).unique()), format_func=friendly_gender, key="pred_gender")
    grade = st.selectbox("Grade", sorted(df["Grade"].dropna().astype(str).unique()), key="pred_grade")
    left, right = st.columns(2)
    overtime = left.number_input("Overtime pay", min_value=0.0, value=0.0, step=250.0, format="%.2f")
    longevity = right.number_input("Longevity pay", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    if st.button("Estimate base salary", type="primary"):
        try:
            estimate = predict_salary({"Department": dept, "Department_Name": department_name, "Division": division, "Gender": gender, "Overtime_Pay": overtime, "Longevity_Pay": longevity, "Grade": grade}, df)
            st.success(f"Estimated base salary  ·  {money(estimate)}")
            st.caption("This is a model estimate, not a guaranteed salary or employment decision.")
        except Exception as exc:
            st.error(f"Prediction could not be prepared: {exc}")


def dataset_page(df: pd.DataFrame | None, source: str | None):
    section("Dataset explorer", "Data understanding")
    if df is None:
        st.info("Run `python clean_data.py` from the project folder to create `data/cleaned_employee_salary.csv`, or upload that cleaned CSV here.")
        st.markdown("**Notebook source:** Employee Salaries · cleaned and preprocessed · 8 original fields")
        return
    a, b, c = st.columns(3)
    a.metric("Dataset shape", f"{df.shape[0]:,} × {df.shape[1]}")
    b.metric("Input features", str(len(FEATURES)))
    c.metric("Target variable", "Base_Salary")
    st.caption(f"Loaded from {source}")
    tab1, tab2, tab3 = st.tabs(["Preview", "Column information", "Missing values"])
    with tab1:
        st.dataframe(df, width="stretch", height=620, hide_index=True)
    with tab2:
        info = pd.DataFrame({"Column": df.columns, "Data type": [str(t) for t in df.dtypes], "Non-null values": df.notna().sum().values, "Unique values": df.nunique(dropna=True).values})
        st.dataframe(info, width="stretch", hide_index=True)
    with tab3:
        missing = df.isna().sum().rename_axis("Column").reset_index(name="Missing values")
        missing["Missing %"] = (missing["Missing values"] / max(len(df), 1) * 100).round(2)
        st.dataframe(missing, width="stretch", hide_index=True)
        st.caption(f"Rows with any missing value: {int(df.isna().any(axis=1).sum()):,}")
    st.markdown("")
    st.download_button(
        "Download cleaned employee CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_employee_salary.csv",
        mime="text/csv",
        type="primary",
    )


def main():
    apply_theme()
    metrics = load_metrics()
    try:
        comparison = pd.read_csv(COMPARISON_PATH)
    except (OSError, pd.errors.ParserError):
        comparison = pd.DataFrame(columns=["Actual Salary", "Predicted Salary"])
    with st.sidebar:
        st.markdown("<div style='font-family:Manrope;font-size:1.12rem;font-weight:800;color:#f0f3ff;padding:.5rem 0'>◈ &nbsp; SALARY<span style='color:#46d8ef'>IQ</span></div>", unsafe_allow_html=True)
        st.caption("EMPLOYEE ANALYTICS · MODEL STUDIO")
        page = st.radio("Workspace", ["Overview", "Salary Analysis", "Model Performance", "Salary Prediction", "Dataset"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**Data source**")
        uploaded = st.file_uploader("Upload cleaned_employee_salary.csv", type=["csv"], label_visibility="collapsed")
        st.caption("Upload the cleaned notebook export. CSV stays in this session.")
        st.markdown("---")
        st.markdown("<span class='tag'>MODEL · LINEAR REGRESSION</span>", unsafe_allow_html=True)
        st.caption("Target: Base Salary")
        st.caption("Evaluation: held-out 20% test set")
    df, source = load_dataset(uploaded)
    if page == "Overview":
        overview(df, metrics, source, comparison)
    elif page == "Salary Analysis":
        salary_analysis(df)
    elif page == "Model Performance":
        model_performance(metrics, comparison)
    elif page == "Salary Prediction":
        prediction_page(df)
    else:
        dataset_page(df, source)
    st.markdown("<div style='text-align:center;color:#65738e;font-size:.75rem;padding:1.8rem 0 .2rem'>EMPLOYEE SALARY INTELLIGENCE &nbsp;·&nbsp; LINEAR REGRESSION</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
