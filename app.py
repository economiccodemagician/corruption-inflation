from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Corruption and Inflation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Corruption and Inflation Dashboard")
st.caption(
    "Interactive country-year analysis. Higher Control of Corruption scores "
    "mean stronger control of corruption."
)

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "outputs" / "analysis_ready_corruption_inflation.csv"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run the notebook export cell first."
        )

    data = pd.read_csv(path)

    required = {
        "country_name", "country_code", "year", "inflation",
        "inflation_log", "region", "income_level", "corruption_score"
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    numeric_columns = [
        "year", "inflation", "inflation_log", "corruption_score"
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=list(required)).copy()
    data["year"] = data["year"].astype(int)

    group_order = [
        "Very weak control",
        "Weak control",
        "Strong control",
        "Very strong control",
    ]

    if "corruption_control_group" not in data.columns:
        data["corruption_control_group"] = pd.qcut(
            data["corruption_score"],
            q=4,
            labels=group_order,
            duplicates="drop",
        )
    else:
        data["corruption_control_group"] = (
            data["corruption_control_group"].astype(str)
        )

    return data


try:
    df = load_data(DATA_PATH)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.sidebar.header("Filters")

min_year = int(df["year"].min())
max_year = int(df["year"].max())

selected_years = st.sidebar.slider(
    "Year range",
    min_year,
    max_year,
    (min_year, max_year),
)

region_options = sorted(df["region"].unique())
selected_regions = st.sidebar.multiselect(
    "Region",
    region_options,
    default=region_options,
)

income_options = sorted(df["income_level"].unique())
selected_income_levels = st.sidebar.multiselect(
    "Income level",
    income_options,
    default=income_options,
)

country_options = sorted(
    df.loc[
        df["region"].isin(selected_regions)
        & df["income_level"].isin(selected_income_levels),
        "country_name",
    ].unique()
)

selected_countries = st.sidebar.multiselect(
    "Country",
    country_options,
    default=[],
    placeholder="All countries",
)

inflation_display = st.sidebar.radio(
    "Inflation scale",
    ["Signed logarithm", "Original percentage"],
)

filtered_df = df[
    df["year"].between(selected_years[0], selected_years[1])
    & df["region"].isin(selected_regions)
    & df["income_level"].isin(selected_income_levels)
].copy()

if selected_countries:
    filtered_df = filtered_df[
        filtered_df["country_name"].isin(selected_countries)
    ].copy()

if filtered_df.empty:
    st.warning("No observations match the selected filters.")
    st.stop()

spearman = filtered_df["corruption_score"].corr(
    filtered_df["inflation"],
    method="spearman",
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Country-year observations", f"{len(filtered_df):,}")
m2.metric("Countries", f"{filtered_df['country_code'].nunique():,}")
m3.metric("Median inflation", f"{filtered_df['inflation'].median():.2f}%")
m4.metric(
    "Spearman correlation",
    "N/A" if pd.isna(spearman) else f"{spearman:.3f}",
)

st.markdown("---")
st.subheader("1. Corruption control and inflation")

if inflation_display == "Signed logarithm":
    y_column = "inflation_log"
    y_label = "Signed logarithm of inflation"
else:
    y_column = "inflation"
    y_label = "Inflation rate (%)"

scatter = px.scatter(
    filtered_df,
    x="corruption_score",
    y=y_column,
    color="region",
    opacity=0.55,
    hover_name="country_name",
    hover_data={
        "country_code": True,
        "year": True,
        "income_level": True,
        "inflation": ":.2f",
        "inflation_log": ":.3f",
        "corruption_score": ":.2f",
        "region": False,
    },
    labels={
        "corruption_score": (
            "Control of corruption score (higher = stronger control)"
        ),
        y_column: y_label,
        "region": "Region",
        "income_level": "Income level",
    },
    title="Relationship between control of corruption and inflation",
)

line_data = filtered_df[
    ["corruption_score", y_column]
].replace([np.inf, -np.inf], np.nan).dropna()

if len(line_data) >= 2 and line_data["corruption_score"].nunique() >= 2:
    slope, intercept = np.polyfit(
        line_data["corruption_score"],
        line_data[y_column],
        1,
    )
    line_x = np.linspace(
        line_data["corruption_score"].min(),
        line_data["corruption_score"].max(),
        100,
    )
    scatter.add_trace(
        go.Scatter(
            x=line_x,
            y=slope * line_x + intercept,
            mode="lines",
            name="Linear trend",
            line=dict(color="black", width=3),
        )
    )

st.plotly_chart(scatter, use_container_width=True)
st.info(
    "A negative correlation means stronger corruption control tends to be "
    "associated with lower inflation. This does not prove causation."
)

st.subheader("2. Median inflation by corruption-control group")

group_order = [
    "Very weak control",
    "Weak control",
    "Strong control",
    "Very strong control",
]

group_summary = (
    filtered_df.groupby(
        "corruption_control_group",
        observed=True,
        as_index=False,
    )
    .agg(
        median_inflation=("inflation", "median"),
        observations=("inflation", "size"),
        countries=("country_code", "nunique"),
    )
)

group_summary["corruption_control_group"] = (
    group_summary["corruption_control_group"].astype(str)
)

group_chart = px.bar(
    group_summary,
    x="corruption_control_group",
    y="median_inflation",
    text_auto=".2f",
    category_orders={"corruption_control_group": group_order},
    hover_data=["observations", "countries"],
    labels={
        "corruption_control_group": "Corruption-control group",
        "median_inflation": "Median inflation rate (%)",
        "observations": "Observations",
        "countries": "Countries",
    },
    title="Median inflation across levels of corruption control",
)
group_chart.update_layout(showlegend=False)
st.plotly_chart(group_chart, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("3. Correlation by region")

    regional = pd.DataFrame(
        [
            {
                "region": region,
                "spearman_correlation": group["corruption_score"].corr(
                    group["inflation"], method="spearman"
                ),
                "observations": len(group),
                "countries": group["country_code"].nunique(),
            }
            for region, group in filtered_df.groupby("region")
            if len(group) >= 3
        ]
    )

    if regional.empty:
        st.warning("Not enough observations for regional correlations.")
    else:
        regional = regional.sort_values("spearman_correlation")
        regional_chart = px.bar(
            regional,
            x="spearman_correlation",
            y="region",
            orientation="h",
            text_auto=".2f",
            hover_data=["observations", "countries"],
            labels={
                "spearman_correlation": "Spearman correlation",
                "region": "Region",
            },
            title="Regional association",
        )
        regional_chart.add_vline(
            x=0,
            line_dash="dash",
            line_color="black",
        )
        regional_chart.update_layout(yaxis_title=None)
        st.plotly_chart(regional_chart, use_container_width=True)

with right:
    st.subheader("4. Correlation over time")

    yearly = pd.DataFrame(
        [
            {
                "year": year,
                "spearman_correlation": group["corruption_score"].corr(
                    group["inflation"], method="spearman"
                ),
                "countries": group["country_code"].nunique(),
            }
            for year, group in filtered_df.groupby("year")
            if len(group) >= 3
        ]
    )

    if yearly.empty:
        st.warning("Not enough observations for yearly correlations.")
    else:
        yearly = yearly.sort_values("year")
        yearly_chart = px.line(
            yearly,
            x="year",
            y="spearman_correlation",
            markers=True,
            hover_data=["countries"],
            labels={
                "year": "Year",
                "spearman_correlation": "Spearman correlation",
            },
            title="Yearly association",
        )
        yearly_chart.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
        )
        st.plotly_chart(yearly_chart, use_container_width=True)

st.subheader("5. Global hierarchy in the latest selected year")

latest_year = int(filtered_df["year"].max())
latest_df = filtered_df[filtered_df["year"] == latest_year].copy()

hierarchy = (
    latest_df.groupby(
        ["region", "income_level", "corruption_control_group"],
        observed=True,
        as_index=False,
    )
    .agg(
        country_count=("country_code", "nunique"),
        median_inflation=("inflation", "median"),
        median_corruption_score=("corruption_score", "median"),
    )
)

hierarchy["corruption_control_group"] = (
    hierarchy["corruption_control_group"].astype(str)
)

treemap = px.treemap(
    hierarchy,
    path=[
        px.Constant("World"),
        "region",
        "income_level",
        "corruption_control_group",
    ],
    values="country_count",
    color="median_inflation",
    color_continuous_scale="RdYlBu_r",
    hover_data={
        "country_count": True,
        "median_inflation": ":.2f",
        "median_corruption_score": ":.2f",
    },
    title=(
        "Inflation by region, income level, and corruption control "
        f"in {latest_year}"
    ),
    labels={
        "country_count": "Countries",
        "median_inflation": "Median inflation (%)",
        "median_corruption_score": (
            "Median corruption-control score"
        ),
    },
)
treemap.update_layout(margin=dict(t=60, l=10, r=10, b=10))
st.plotly_chart(treemap, use_container_width=True)

with st.expander("View and download filtered data"):
    display_columns = [
        "country_name",
        "country_code",
        "year",
        "region",
        "income_level",
        "inflation",
        "inflation_log",
        "corruption_score",
        "corruption_control_group",
    ]

    st.dataframe(
        filtered_df[display_columns].sort_values(
            ["year", "country_name"],
            ascending=[False, True],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_corruption_inflation.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Sources: World Bank World Development Indicators and Worldwide "
    "Governance Indicators. Descriptive association only; no causal claim."
)
