"""
analytics.py
Builds Plotly charts from violation data in the database: type distribution,
zone-wise risk ranking (the "where to prioritize enforcement" answer),
hourly trend heatmap, and summary metrics. Pulls data via database.py
query functions — no direct SQL here, keeps concerns separated.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import database


def violation_type_pie_chart():
    """
    Pie chart of violation counts by type.
    Returns a Plotly figure, or None if there's no data yet.
    """
    counts = database.get_violation_counts_by_type()
    if not counts:
        return None

    labels = [k.replace("_", " ").title() for k in counts.keys()]
    values = list(counts.values())

    fig = px.pie(
        names=labels,
        values=values,
        title="Violations by Type",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def zone_risk_bar_chart():
    """
    Bar chart ranking zones by total severity score (not just raw count),
    so it directly answers "where should enforcement be prioritized."
    Returns a Plotly figure, or None if there's no data yet.
    """
    zone_data = database.get_violation_counts_by_zone()
    if not zone_data:
        return None

    df = pd.DataFrame(zone_data)
    df = df.sort_values("total_severity", ascending=True)  # ascending for horizontal bar order

    fig = px.bar(
        df,
        x="total_severity",
        y="zone",
        orientation="h",
        title="Enforcement Priority by Zone (Risk-Weighted)",
        labels={"total_severity": "Total Severity Score", "zone": "Zone"},
        color="total_severity",
        color_continuous_scale="Reds",
        text="count",
    )
    fig.update_traces(texttemplate="%{text} violations", textposition="outside")
    fig.update_layout(margin=dict(t=50, b=20, l=20, r=20), coloraxis_showscale=False)
    return fig


def hourly_trend_chart():
    """
    Bar chart of violation counts by hour of day — shows enforcement
    teams which hours need the most coverage.
    """
    hour_counts = database.get_violations_by_hour()
    if not any(hour_counts.values()):
        return None

    df = pd.DataFrame({
        "hour": [f"{h:02d}:00" for h in hour_counts.keys()],
        "count": list(hour_counts.values()),
    })

    fig = px.bar(
        df,
        x="hour",
        y="count",
        title="Violations by Hour of Day",
        labels={"hour": "Hour", "count": "Violation Count"},
        color="count",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(margin=dict(t=50, b=20, l=20, r=20), coloraxis_showscale=False)
    return fig


def severity_gauge(avg_severity):
    """
    A gauge chart showing average severity across all logged violations —
    a quick-glance dashboard widget for overall risk level.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_severity,
        title={"text": "Average Violation Severity"},
        gauge={
            "axis": {"range": [0, 10]},
            "bar": {"color": "#e74c3c" if avg_severity >= 7 else
                             "#f39c12" if avg_severity >= 4 else "#2ecc71"},
            "steps": [
                {"range": [0, 4], "color": "#d4f7d4"},
                {"range": [4, 7], "color": "#fde8c8"},
                {"range": [7, 10], "color": "#f7d4d4"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def repeat_offenders_table(min_violations=3):
    """
    Returns a pandas DataFrame of repeat offenders for display in a
    Streamlit table — plates with min_violations or more logged.
    """
    offenders = database.get_repeat_offenders(min_violations)
    if not offenders:
        return pd.DataFrame(columns=["Plate Number", "Violation Count"])

    df = pd.DataFrame(offenders)
    df.columns = ["Plate Number", "Violation Count"]
    return df


def evidence_log_table(limit=100):
    """
    Returns a pandas DataFrame of recent violations formatted for display
    in the Evidence Log tab.
    """
    records = database.get_all_violations(limit)
    if not records:
        return pd.DataFrame(columns=[
            "Time", "Type", "Plate", "Zone", "Confidence", "Severity", "Source"
        ])

    df = pd.DataFrame(records)
    display_df = pd.DataFrame({
        "Time": df["timestamp"],
        "Type": df["violation_type"].str.replace("_", " ").str.title(),
        "Plate": df["plate_number"].fillna("Unidentified"),
        "Zone": df["zone"].fillna("Unspecified"),
        "Confidence": (df["confidence"] * 100).round(1).astype(str) + "%",
        "Severity": df["severity"],
        "Source": df["source"],
    })
    return display_df


if __name__ == "__main__":
    # Standalone test — requires violations.db to have some data.
    # Run database.py first if you haven't, to populate test records.
    database.init_db()

    print("Testing analytics functions...\n")

    pie = violation_type_pie_chart()
    print(f"Pie chart: {'generated' if pie else 'no data yet'}")

    bar = zone_risk_bar_chart()
    print(f"Zone risk bar chart: {'generated' if bar else 'no data yet'}")

    hourly = hourly_trend_chart()
    print(f"Hourly trend chart: {'generated' if hourly else 'no data yet'}")

    stats = database.get_summary_stats()
    print(f"\nSummary stats: {stats}")

    gauge = severity_gauge(stats["avg_severity"])
    print(f"Severity gauge: {'generated' if gauge else 'no data'}")

    offenders_df = repeat_offenders_table(min_violations=2)
    print(f"\nRepeat offenders table:\n{offenders_df}")

    log_df = evidence_log_table()
    print(f"\nEvidence log table:\n{log_df}")

    print("\nIf you see 'no data yet' for charts, run 'python database.py' first")
    print("to insert test records, then re-run this script.")