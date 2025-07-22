import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ydata_profiling
import streamlit.components.v1 as components

# === PAGE CONFIG ===
st.set_page_config(
    page_title="InsightX – Smart Automated EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f7;
    }
    h1, h2, h3 {
        color: #222831;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR FILE UPLOAD ===
st.sidebar.title("📁 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# === INIT SESSION STATE ===
if uploaded_file is not None:
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = pd.read_csv(uploaded_file)
    df = st.session_state.cleaned_df.copy()

    if "history" not in st.session_state:
        st.session_state.history = []

    # === SIDEBAR FILTERS ===
    st.sidebar.header("🔎 Filters")
    df_filtered = df.copy()

    # Numeric Filter
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        col_to_filter = st.sidebar.selectbox("Numeric Column", numeric_cols)
        min_val, max_val = float(df[col_to_filter].min()), float(df[col_to_filter].max())
        selected_range = st.sidebar.slider(
            f"{col_to_filter} range",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        df_filtered = df_filtered[(df_filtered[col_to_filter] >= selected_range[0]) &
                                  (df_filtered[col_to_filter] <= selected_range[1])]

    # Categorical Filter
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        cat_col = st.sidebar.selectbox("Categorical Column", categorical_cols)
        unique_cats = df[cat_col].unique().tolist()
        selected_cats = st.sidebar.multiselect(f"{cat_col} values", unique_cats, default=unique_cats)
        df_filtered = df_filtered[df_filtered[cat_col].isin(selected_cats)]

    # Save session history
    if uploaded_file.name not in [h["filename"] for h in st.session_state.history]:
        st.session_state.history.insert(0, {
            "filename": uploaded_file.name,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "nulls": df.isnull().sum().to_dict()
        })
        if len(st.session_state.history) > 5:
            st.session_state.history.pop()

    st.sidebar.success(f"✅ File uploaded! Filtered rows: {df_filtered.shape[0]}")

    # === MAIN TABS ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗂️ Overview",
        "📊 Visuals",
        "🧹 Outliers & Missing",
        "📈 Profiling Report",
        "💾 Download"
    ])

    # === OVERVIEW ===
    with tab1:
        st.markdown("<h1 style='text-align: center;'>📊 InsightX – Smart Automated EDA</h1>", unsafe_allow_html=True)
        st.write(f"Filtered dataset shape: {df_filtered.shape}")
        st.dataframe(df_filtered, use_container_width=True)

        st.subheader("📌 Basic Info")
        st.write(f"Original: {df.shape} | Filtered: {df_filtered.shape}")
        st.write("Columns:", df.columns.tolist())
        st.write("Nulls:", df.isnull().sum())

        st.subheader("💡 Meaningful Insights")
        insights = []

        # High Correlations
        if len(numeric_cols) >= 2:
            corr_matrix = df_filtered[numeric_cols].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7 and abs(corr_val) < 1:
                        insights.append(f"📈 {corr_matrix.columns[i]} is strongly correlated with {corr_matrix.columns[j]} (r = {corr_val:.2f})")

        # Dominant Category
        for cat_col in categorical_cols:
            if cat_col in df_filtered.columns:
                top_cat = df_filtered[cat_col].value_counts(normalize=True).idxmax()
                share = df_filtered[cat_col].value_counts(normalize=True).max() * 100
                if share > 30:
                    insights.append(f"🏷️ {top_cat} is dominant in {cat_col} ({share:.1f}%)")

        # Null Percentages
        nulls = df_filtered.isnull().mean()
        for col, pct in nulls.items():
            if pct > 0:
                insights.append(f"⚠️ {col} has {pct*100:.1f}% missing values")

        # Outlier %
        for col in numeric_cols:
            Q1 = df_filtered[col].quantile(0.25)
            Q3 = df_filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df_filtered[(df_filtered[col] < Q1 - 1.5 * IQR) | (df_filtered[col] > Q3 + 1.5 * IQR)]
            pct_outliers = len(outliers) / len(df_filtered) * 100
            if pct_outliers > 5:
                insights.append(f"🚩 {col} has {pct_outliers:.1f}% outliers")

        # Trend Example
        if 'work_year' in df_filtered.columns and 'salary_in_usd' in df_filtered.columns:
            trend = df_filtered.groupby('work_year')['salary_in_usd'].mean()
            if trend.shape[0] >= 2:
                growth = ((trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0]) * 100
                insights.append(f"📊 Average salary changed by {growth:.1f}% from {trend.index[0]} to {trend.index[-1]}")

        if insights:
            for ins in insights:
                st.write(ins)
        else:
            st.write("✅ No major patterns found.")

        st.session_state.generated_insights = insights

    # === VISUALS ===
    with tab2:
        st.header("📊 Visuals on Filtered Data")
        if df_filtered.empty:
            st.warning("⚠️ No data after filters.")
        else:
            chart_type = st.selectbox(
                "Select Chart",
                ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap", "Line Plot", "Pie Chart"]
            )

            if chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
                x_axis = st.selectbox("X-axis", numeric_cols)
                y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
                fig = px.scatter(df_filtered, x=x_axis, y=y_axis, trendline="ols")
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Histogram" and numeric_cols:
                col = st.selectbox("Column", numeric_cols)
                fig = px.histogram(df_filtered, x=col, nbins=30)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Box Plot" and numeric_cols:
                col = st.selectbox("Column", numeric_cols)
                fig = px.box(df_filtered, y=col)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Correlation Heatmap" and len(numeric_cols) >= 2:
                corr = df_filtered[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Line Plot" and numeric_cols:
                y_col = st.selectbox("Y-axis", numeric_cols)
                x_col = st.selectbox("X-axis", df_filtered.columns.tolist())
                fig = px.line(df_filtered, x=x_col, y=y_col)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Pie Chart" and categorical_cols:
                cat_col = st.selectbox("Category Column", categorical_cols)
                counts = df_filtered[cat_col].value_counts()
                fig = px.pie(names=counts.index, values=counts.values)
                st.plotly_chart(fig, use_container_width=True)

    # === OUTLIERS & MISSING ===
    with tab3:
        st.header("🧹 Outlier Detection & Missing Value Treatment")
        out_col = st.selectbox("Column for Outlier Detection", numeric_cols)

        Q1 = df[out_col].quantile(0.25)
        Q3 = df[out_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[out_col] < lower) | (df[out_col] > upper)]
        st.write(f"Outliers found: {outliers.shape[0]}")
        st.dataframe(outliers)

        if st.button("Drop Outliers"):
            st.session_state.cleaned_df = df[(df[out_col] >= lower) & (df[out_col] <= upper)]
            st.success(f"Outliers removed! New shape: {st.session_state.cleaned_df.shape}")

        st.subheader("🩹 Missing Value Treatment")
        nulls = df.isnull().sum()
        st.write(nulls)

        if nulls.sum() > 0:
            fill_col = st.selectbox("Column to fill nulls", df.columns[df.isnull().sum() > 0])
            method = st.radio("Method", ["Mean", "Median", "Mode"])
            if st.button("Fill Nulls"):
                if method == "Mean":
                    val = df[fill_col].mean()
                elif method == "Median":
                    val = df[fill_col].median()
                else:
                    val = df[fill_col].mode()[0]
                df[fill_col].fillna(val, inplace=True)
                st.session_state.cleaned_df = df.copy()
                st.success(f"Filled nulls in {fill_col} with {method}: {val:.2f}")
        else:
            st.info("✅ No missing values.")

    # === PROFILING ===
    with tab4:
        st.header("📈 Profiling Report")
        if st.button("Generate Profiling"):
            profile = ydata_profiling.ProfileReport(df_filtered, title="Profiling", explorative=True)
            html = profile.to_html()
            components.html(html, height=800, scrolling=True)
            st.session_state.profile_html = html

    # === DOWNLOAD ===
    with tab5:
        st.header("💾 Download Files")
        st.download_button(
            "Download CSV",
            df_filtered.to_csv(index=False).encode('utf-8'),
            "filtered_data.csv",
            "text/csv"
        )

        st.subheader("📝 Download Data Report (TXT)")
        report_lines = [
            f"📁 Filename: {uploaded_file.name}",
            f"📐 Shape: {df_filtered.shape[0]} rows × {df_filtered.shape[1]} columns",
            f"🧾 Columns: {', '.join(df_filtered.columns)}",
            "❗ Null Values Per Column:"
        ]
        nulls = df_filtered.isnull().sum()
        for col, val in nulls.items():
            report_lines.append(f"  - {col}: {val}")
        report_lines.append("\n💡 Insights:")
        if st.session_state.get("generated_insights"):
            for ins in st.session_state.generated_insights:
                report_lines.append(f"- {ins}")
        else:
            report_lines.append("- No major patterns found.")
        report_txt = "\n".join(report_lines)
        st.download_button(
            "Download Data Report (TXT)",
            report_txt,
            "data_report.txt",
            "text/plain"
        )

        st.subheader("📝 Download Data Report (HTML)")
        report_html = f"""
        <!DOCTYPE html>
        <html><head><meta charset="UTF-8"><title>InsightX – Data Report</title></head>
        <body><h1>📊 InsightX – Data Report</h1>
        <h2>📁 Filename:</h2><pre>{uploaded_file.name}</pre>
        <h2>📐 Shape:</h2><pre>{df_filtered.shape[0]} rows × {df_filtered.shape[1]} columns</pre>
        <h2>🧾 Columns:</h2><pre>{', '.join(df_filtered.columns)}</pre>
        <h2>❗ Null Values Per Column:</h2><pre>
        """
        for col, val in nulls.items():
            report_html += f"  - {col}: {val}\n"
        report_html += "</pre><h2>💡 Insights:</h2><pre>"
        if st.session_state.get("generated_insights"):
            for ins in st.session_state.generated_insights:
                report_html += f"- {ins}\n"
        else:
            report_html += "- No major patterns found.\n"
        report_html += "</pre></body></html>"
        st.download_button(
            "Download Data Report (HTML)",
            report_html.encode('utf-8'),
            "data_report.html",
            "text/html"
        )

        if "profile_html" in st.session_state:
            st.subheader("📈 Profiling Report")
            st.download_button(
                "Download Profiling Report",
                st.session_state.profile_html.encode('utf-8'),
                "profiling_report.html",
                "text/html"
            )
        else:
            st.info("Generate profiling report first to download.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🕘 Previous Sessions")
    for h in st.session_state.history:
        with st.sidebar.expander(f"📁 {h['filename']}"):
            st.write(f"Shape: {h['shape']}")
            st.write(f"Columns: {h['columns']}")
            st.write("Nulls:", h['nulls'])

else:
    st.warning("📤 Please upload a CSV file to get started!")


















