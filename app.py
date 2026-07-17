from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Corruption and Inflation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Corruption and Inflation Dashboard")
st.caption(
    "Interactive country-year analysis using World Bank inflation and "
    "Control of Corruption data. Higher corruption-control scores mean "
    "stronger control of corruption."
)

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "outputs" / "analysis_ready_corruption_inflation.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Make sure the CSV is inside the "
            "outputs folder."
        )

    data = pd.read_csv(path)

    required_columns = {
        "country_name",
        "country_code",
        "year",
        "inflation",
        "inflation_log",
        "region",
        "income_level",
        "corruption_score",
    }

    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The dashboard CSV is missing: "
            + ", ".join(sorted(missing_columns))
        )

    numeric_columns = [
        "year",
        "inflation",
        "inflation_log",
        "corruption_score",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=list(required_columns)).copy()
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


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Filters")

minimum_year = int(df["year"].min())
maximum_year = int(df["year"].max())

selected_years = st.sidebar.slider(
    "Year range",
    minimum_year,
    maximum_year,
    (minimum_year, maximum_year),
)

region_options = sorted(df["region"].dropna().unique())
selected_regions = st.sidebar.multiselect(
    "Region",
    region_options,
    default=region_options,
)

income_options = sorted(df["income_level"].dropna().unique())
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
    ].dropna().unique()
)

selected_countries = st.sidebar.multiselect(
    "Country",
    country_options,
    default=[],
    placeholder="All countries",
)

inflation_display = st.sidebar.radio(
    "Inflation scale for scatter plot",
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


# ---------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Question 1
# ---------------------------------------------------------
st.header("Question 1: Is stronger corruption control associated with lower inflation?")
st.write(
    "This scatter plot compares the Control of Corruption score with inflation. "
    "The line shows the general direction of the relationship."
)

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
    title="Control of corruption and inflation",
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

if pd.isna(spearman):
    st.info("Not enough variation is available to calculate the correlation.")
elif spearman < 0:
    st.success(
        f"Answer: The filtered data show a negative association "
        f"(Spearman correlation = {spearman:.3f}). Stronger corruption "
        "control tends to be associated with lower inflation."
    )
else:
    st.info(
        f"Answer: The filtered data do not show a negative association "
        f"(Spearman correlation = {spearman:.3f})."
    )

st.caption("The result is descriptive and does not prove causation.")
st.markdown("---")


# ---------------------------------------------------------
# Question 2
# ---------------------------------------------------------
st.header("Question 2: How does median inflation differ across corruption-control groups?")
st.write(
    "The observations are divided into four corruption-control groups. "
    "Median inflation is used because it is less affected by hyperinflation."
)

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
    title="Median inflation by corruption-control group",
)

group_chart.update_layout(showlegend=False)
st.plotly_chart(group_chart, use_container_width=True)

if not group_summary.empty:
    highest_group = group_summary.loc[
        group_summary["median_inflation"].idxmax()
    ]
    lowest_group = group_summary.loc[
        group_summary["median_inflation"].idxmin()
    ]

    st.success(
        f"Answer: The highest median inflation is in the "
        f"{highest_group['corruption_control_group']} group "
        f"({highest_group['median_inflation']:.2f}%), while the lowest is in "
        f"the {lowest_group['corruption_control_group']} group "
        f"({lowest_group['median_inflation']:.2f}%)."
    )

st.markdown("---")


# ---------------------------------------------------------
# Question 3 — G20 multiple-line chart
# ---------------------------------------------------------
st.header(
    "Question 3: How has the corruption–inflation relationship changed "
    "among G20 countries?"
)
st.write(
    "For each year, the chart calculates the relationship between inflation "
    "and corruption control across the available G20 countries. It compares "
    "Spearman correlation, Pearson correlation using original inflation, and "
    "Pearson correlation using signed-log inflation."
)
st.caption(
    "This chart shows correlation between corruption control and inflation in the G20 countries over the years it uses spearman and pearson correlation."
    ""
    ""
)

g20_codes = [
    "ARG", "AUS", "BRA", "CAN", "CHN",
    "FRA", "DEU", "IND", "IDN", "ITA",
    "JPN", "KOR", "MEX", "RUS", "SAU",
    "ZAF", "TUR", "GBR", "USA",
]

g20_df = df[
    df["country_code"].isin(g20_codes)
    & df["year"].between(selected_years[0], selected_years[1])
].copy()

g20_yearly_correlations = pd.DataFrame(
    [
        {
            "year": year,
            "Spearman": group["corruption_score"].corr(
                group["inflation"],
                method="spearman",
            ),
            "Pearson — original inflation": group[
                "corruption_score"
            ].corr(
                group["inflation"],
                method="pearson",
            ),
            "Pearson — signed-log inflation": group[
                "corruption_score"
            ].corr(
                group["inflation_log"],
                method="pearson",
            ),
            "countries": group["country_code"].nunique(),
        }
        for year, group in g20_df.groupby("year")
        if group["country_code"].nunique() >= 15
    ]
)

if g20_yearly_correlations.empty:
    st.warning(
        "The selected year range does not contain enough G20 countries "
        "to calculate reliable yearly correlations."
    )
else:
    g20_correlations_long = g20_yearly_correlations.melt(
        id_vars=["year", "countries"],
        value_vars=[
            "Spearman",
            "Pearson — original inflation",
            "Pearson — signed-log inflation",
        ],
        var_name="correlation_measure",
        value_name="correlation",
    ).dropna(subset=["correlation"])

    g20_chart = px.line(
        g20_correlations_long,
        x="year",
        y="correlation",
        color="correlation_measure",
        markers=True,
        hover_data={
            "countries": True,
            "correlation": ":.3f",
            "correlation_measure": False,
        },
        title=(
            "Yearly corruption–inflation correlations among G20 countries"
        ),
        labels={
            "year": "Year",
            "correlation": "Correlation coefficient",
            "correlation_measure": "Correlation measure",
            "countries": "G20 countries included",
        },
    )

    g20_chart.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
    )

    g20_chart.update_yaxes(range=[-1, 1])

    g20_chart.update_layout(
        legend_title_text="Correlation measure",
        hovermode="x unified",
    )

    st.plotly_chart(g20_chart, use_container_width=True)

    spearman_results = g20_yearly_correlations.dropna(
        subset=["Spearman"]
    )

    if not spearman_results.empty:
        strongest_g20_year = spearman_results.loc[
            spearman_results["Spearman"].idxmin()
        ]
        weakest_g20_year = spearman_results.loc[
            spearman_results["Spearman"].abs().idxmin()
        ]

        st.success(
            f"Answer: The strongest negative G20 Spearman relationship "
            f"within the selected period occurs in "
            f"{int(strongest_g20_year['year'])} "
            f"({strongest_g20_year['Spearman']:.3f}). "
            f"The relationship closest to zero occurs in "
            f"{int(weakest_g20_year['year'])} "
            f"({weakest_g20_year['Spearman']:.3f})."
        )

    st.caption(
        "Differences between the Pearson lines show how strongly extreme "
        "inflation observations affect the measured relationship. The chart "
        "shows association across G20 countries, not causation."
    )

st.markdown("---")


# ---------------------------------------------------------
# Questions 4 and 5
# ---------------------------------------------------------
left_column, right_column = st.columns(2)

with left_column:
    st.header("Question 4: Does the relationship differ by region?")
    st.write(
        "A separate Spearman correlation is calculated for each selected region."
    )

    regional_correlations = pd.DataFrame(
        [
            {
                "region": region,
                "spearman_correlation": group[
                    "corruption_score"
                ].corr(group["inflation"], method="spearman"),
                "observations": len(group),
                "countries": group["country_code"].nunique(),
            }
            for region, group in filtered_df.groupby("region")
            if len(group) >= 3
        ]
    )

    if regional_correlations.empty:
        st.warning("Not enough observations for regional correlations.")
    else:
        regional_correlations = regional_correlations.dropna(
            subset=["spearman_correlation"]
        ).sort_values("spearman_correlation")

        regional_chart = px.bar(
            regional_correlations,
            x="spearman_correlation",
            y="region",
            orientation="h",
            text_auto=".2f",
            hover_data=["observations", "countries"],
            labels={
                "spearman_correlation": "Spearman correlation",
                "region": "Region",
                "observations": "Observations",
                "countries": "Countries",
            },
            title="Regional corruption–inflation association",
        )

        regional_chart.add_vline(
            x=0,
            line_dash="dash",
            line_color="black",
        )
        regional_chart.update_layout(yaxis_title=None)
        st.plotly_chart(regional_chart, use_container_width=True)

        if not regional_correlations.empty:
            strongest_region = regional_correlations.iloc[0]
            weakest_region = regional_correlations.iloc[
                regional_correlations[
                    "spearman_correlation"
                ].abs().argmin()
            ]

            st.success(
                f"Answer: The strongest negative association is in "
                f"{strongest_region['region']} "
                f"({strongest_region['spearman_correlation']:.3f}). "
                f"The relationship closest to zero is in "
                f"{weakest_region['region']} "
                f"({weakest_region['spearman_correlation']:.3f})."
            )

with right_column:
    st.header("Question 5: Has the relationship changed over time?")
    st.write(
        "A separate Spearman correlation is calculated for each selected year."
    )

    yearly_correlations = pd.DataFrame(
        [
            {
                "year": year,
                "spearman_correlation": group[
                    "corruption_score"
                ].corr(group["inflation"], method="spearman"),
                "countries": group["country_code"].nunique(),
            }
            for year, group in filtered_df.groupby("year")
            if len(group) >= 3
        ]
    )

    if yearly_correlations.empty:
        st.warning("Not enough observations for yearly correlations.")
    else:
        yearly_correlations = yearly_correlations.dropna(
            subset=["spearman_correlation"]
        ).sort_values("year")

        yearly_chart = px.line(
            yearly_correlations,
            x="year",
            y="spearman_correlation",
            markers=True,
            hover_data=["countries"],
            labels={
                "year": "Year",
                "spearman_correlation": "Spearman correlation",
                "countries": "Countries",
            },
            title="Yearly corruption–inflation association",
        )

        yearly_chart.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
        )

        st.plotly_chart(yearly_chart, use_container_width=True)

        if not yearly_correlations.empty:
            strongest_year = yearly_correlations.loc[
                yearly_correlations["spearman_correlation"].idxmin()
            ]
            weakest_year = yearly_correlations.loc[
                yearly_correlations["spearman_correlation"].abs().idxmin()
            ]

            st.success(
                f"Answer: The strongest negative yearly association occurs in "
                f"{int(strongest_year['year'])} "
                f"({strongest_year['spearman_correlation']:.3f}). "
                f"The relationship closest to zero occurs in "
                f"{int(weakest_year['year'])} "
                f"({weakest_year['spearman_correlation']:.3f})."
            )

st.markdown("---")


# ---------------------------------------------------------
# Question 6
# ---------------------------------------------------------
st.header("Question 6: Where is high inflation concentrated in the global hierarchy?")
st.write(
    "The treemap groups countries by region, income level, and "
    "corruption-control group in the latest selected year. Rectangle size "
    "represents the number of countries, while color represents median inflation."
)

latest_selected_year = int(filtered_df["year"].max())
latest_df = filtered_df[
    filtered_df["year"] == latest_selected_year
].copy()

hierarchy_data = (
    latest_df.groupby(
        [
            "region",
            "income_level",
            "corruption_control_group",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        country_count=("country_code", "nunique"),
        median_inflation=("inflation", "median"),
        median_corruption_score=("corruption_score", "median"),
    )
)

hierarchy_data["corruption_control_group"] = (
    hierarchy_data["corruption_control_group"].astype(str)
)

treemap = px.treemap(
    hierarchy_data,
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
        f"in {latest_selected_year}"
    ),
    labels={
        "country_count": "Countries",
        "median_inflation": "Median inflation (%)",
        "median_corruption_score": (
            "Median corruption-control score"
        ),
        "income_level": "Income level",
        "corruption_control_group": "Corruption-control group",
    },
)

treemap.update_layout(margin=dict(t=60, l=10, r=10, b=10))
st.plotly_chart(treemap, use_container_width=True)

if not hierarchy_data.empty:
    highest_hierarchy_group = hierarchy_data.loc[
        hierarchy_data["median_inflation"].idxmax()
    ]

    st.success(
        f"Answer: In {latest_selected_year}, the highest median inflation "
        f"among the displayed groups is found in "
        f"{highest_hierarchy_group['region']} / "
        f"{highest_hierarchy_group['income_level']} / "
        f"{highest_hierarchy_group['corruption_control_group']} "
        f"({highest_hierarchy_group['median_inflation']:.2f}%)."
    )


# ---------------------------------------------------------
# Data table and download
# ---------------------------------------------------------
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
    "Governance Indicators. Results are descriptive associations and do not "
    "establish causation."
)
