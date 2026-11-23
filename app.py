
import os
import hashlib
import joblib
import psutil
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

try:
    from google_play_scraper import search as play_search
    from google_play_scraper import app as play_app
except ImportError:
    play_search = None
    play_app = None


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Automated NIDS",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "models/nids_model.pkl"
BENIGN_NAMES = ["BENIGN", "NORMAL", "NORMAL TRAFFIC", "LEGITIMATE"]
CHART_MAX_POINTS = 3000


# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>
.stApp { background:#071426; color:#f3f7ff; }
[data-testid="stSidebar"] { background:#0c1d33; }
[data-testid="stSidebar"] * { color:white !important; }
.block-container { padding:2rem 2.5rem; max-width:1800px; }
h1,h2,h3 { color:#ffffff !important; }
p,label { color:#dcecff !important; }

.metricbox {
    background:linear-gradient(135deg,#173b60,#10243e);
    border:1px solid #3474a8;
    padding:22px;
    border-radius:16px;
    min-height:135px;
}
.metriclabel { color:#b9dcff; font-size:16px; font-weight:700; }
.metricvalue { color:white; font-size:34px; font-weight:800; margin-top:8px; }
.metricsub { color:#a9c9e8; font-size:13px; margin-top:6px; }

div[data-testid="stMetric"] {
    background:#153453 !important;
    border:1px solid #3474a8 !important;
    border-radius:12px !important;
    padding:16px !important;
}
div[data-testid="stMetricLabel"] * { color:#c5e3ff !important; }
div[data-testid="stMetricValue"] * { color:white !important; }

[data-testid="stFileUploader"] {
    background:#102945;
    border:1px solid #3474a8;
    border-radius:12px;
    padding:12px;
}
[data-testid="stFileUploader"] * { color:white !important; }
.stButton button,.stDownloadButton button {
    min-height:44px;
    border-radius:9px;
    font-weight:700;
}
[data-testid="stDataFrame"] {
    border:1px solid #3474a8;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "results": None,
    "report": None,
    "results_upload_key": None,
    "active_upload_key": None,
    "network_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# HELPERS
# =====================================================

def metric_card(title, value, subtitle, color="#ffffff"):
    st.markdown(f"""
    <div class="metricbox">
        <div class="metriclabel">{title}</div>
        <div class="metricvalue" style="color:{color}">{value}</div>
        <div class="metricsub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def is_benign(label):
    return str(label).strip().upper() in BENIGN_NAMES


def load_model_data():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


def prepare_features(df, model_data):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    features = model_data["features"]
    missing = [f for f in features if f not in df.columns]

    X = pd.DataFrame(index=df.index)

    for feature in features:
        if feature in df.columns:
            X[feature] = pd.to_numeric(df[feature], errors="coerce")
        else:
            X[feature] = 0

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, missing


def predict_dataset(df):
    model_data = load_model_data()
    if model_data is None:
        return None

    try:
        model = model_data["model"]
        encoder = model_data["label_encoder"]

        X, missing = prepare_features(df, model_data)

        predicted_encoded = model.predict(X)
        predicted_labels = encoder.inverse_transform(
            np.asarray(predicted_encoded).astype(int)
        )

        # Prediction confidence
        confidence = np.full(len(predicted_labels), np.nan)

        # Initial attack probability fallback
        attack_probability = np.array([
            0 if is_benign(label) else 100
            for label in predicted_labels
        ], dtype=float)

        # Use actual model probabilities when available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            confidence = probabilities.max(axis=1) * 100

            model_labels = []
            for class_id in model.classes_:
                try:
                    decoded = encoder.inverse_transform([int(class_id)])[0]
                    model_labels.append(str(decoded).strip())
                except Exception:
                    model_labels.append(str(class_id))

            benign_indices = [
                i for i, label in enumerate(model_labels)
                if is_benign(label)
            ]

            if benign_indices:
                benign_probability = probabilities[:, benign_indices].sum(axis=1)
                attack_probability = (1 - benign_probability) * 100

        attack_probability = np.clip(attack_probability, 0, 100)

        output = pd.DataFrame({
            "Record ID": np.arange(1, len(df) + 1),
            "Predicted Attack": predicted_labels,
            "Attack Probability (%)": np.round(attack_probability, 2),
            "Prediction Confidence (%)": np.round(confidence, 2)
        })

        output["Status"] = output["Predicted Attack"].apply(
            lambda label: "Normal" if is_benign(label) else "Attack"
        )

        output["Risk Level"] = pd.cut(
            output["Attack Probability (%)"],
            bins=[-0.01, 39.99, 69.99, 100.01],
            labels=["Low", "Medium", "High"]
        ).astype(str)

        # Source and prediction report columns kept distinct
        source = df.copy().reset_index(drop=True)
        source.columns = [f"CSV_{str(c).strip()}" for c in source.columns]

        result_columns = output.copy().reset_index(drop=True)
        result_columns.columns = [f"NIDS_{c}" for c in result_columns.columns]

        report = pd.concat([source, result_columns], axis=1)
        report = report.loc[:, ~report.columns.duplicated()].copy()

        return {
            "output": output,
            "report": report,
            "missing": missing
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# =====================================================
# CHART HELPERS
# =====================================================

def donut_chart(labels, values, title, colors):
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.60,
        marker=dict(
            colors=colors,
            line=dict(color="#102945", width=3)
        ),
        textinfo="percent",
        textposition="inside",
        textfont=dict(color="white", size=15),
        sort=False,
        hovertemplate="<b>%{label}</b><br>Records: %{value:,}<br>%{percent}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20, color="white")),
        height=380,
        paper_bgcolor="#102945",
        plot_bgcolor="#102945",
        font=dict(color="white"),
        legend=dict(
            orientation="h", y=-0.12, x=0.5,
            xanchor="center", font=dict(color="white", size=12)
        ),
        margin=dict(t=65, b=55, l=15, r=15)
    )
    return fig


def show_analytics(output):
    total = len(output)
    attacks = int((output["Status"] == "Attack").sum())
    normal = total - attacks

    risk_counts = (
        output["Risk Level"].value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )
    low, medium, high = [
        int(risk_counts[x]) for x in ["Low", "Medium", "High"]
    ]

    attack_rate = attacks / total * 100 if total else 0
    normal_rate = normal / total * 100 if total else 0

    st.markdown("---")
    st.header("📊 Security Analytics")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.plotly_chart(
            donut_chart(
                ["Normal", "Attack"], [normal, attacks],
                "Traffic Status", ["#20c997", "#ff5364"]
            ),
            use_container_width=True
        )

    with c2:
        st.plotly_chart(
            donut_chart(
                ["Low", "Medium", "High"], [low, medium, high],
                "Risk Levels", ["#20c997", "#ffb347", "#ff5364"]
            ),
            use_container_width=True
        )

    with c3:
        st.plotly_chart(
            donut_chart(
                ["Attack Rate", "Normal Rate"], [attack_rate, normal_rate],
                "Attack Rate", ["#ff5364", "#20c997"]
            ),
            use_container_width=True
        )

    # Attack types
    st.header("📈 Attack Type Analysis")
    attack_rows = output[output["Status"] == "Attack"]

    if not attack_rows.empty:
        attack_counts = attack_rows["Predicted Attack"].value_counts().reset_index()
        attack_counts.columns = ["Attack Type", "Records"]

        fig = px.bar(
            attack_counts,
            x="Attack Type",
            y="Records",
            color="Attack Type",
            text="Records",
            color_discrete_sequence=[
                "#ff5364", "#ffb347", "#8b7cff",
                "#00c8ff", "#20c997", "#e879f9"
            ]
        )
        fig.update_traces(textposition="outside", textfont=dict(color="white", size=14))
        fig.update_layout(
            height=420,
            paper_bgcolor="#102945",
            plot_bgcolor="#102945",
            font=dict(color="white", size=14),
            showlegend=False,
            margin=dict(t=35, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No attack records detected in this dataset.")

    # Risk distribution
    st.header("🎯 Risk Distribution")

    risk_df = pd.DataFrame({
        "Risk": ["Low", "Medium", "High"],
        "Records": [low, medium, high]
    })

    risk_fig = px.bar(
        risk_df,
        x="Risk",
        y="Records",
        color="Risk",
        text="Records",
        color_discrete_map={
            "Low": "#20c997",
            "Medium": "#ffb347",
            "High": "#ff5364"
        }
    )
    risk_fig.update_traces(textposition="outside", textfont=dict(color="white", size=15))
    risk_fig.update_layout(
        height=380,
        paper_bgcolor="#102945",
        plot_bgcolor="#102945",
        font=dict(color="white", size=14),
        showlegend=False
    )
    st.plotly_chart(risk_fig, use_container_width=True)

    # Sample chart points only, full predictions remain intact
    st.header("📉 Attack Probability Overview")
    if len(output) > CHART_MAX_POINTS:
        chart_data = output.sample(CHART_MAX_POINTS, random_state=42).sort_values("Record ID")
    else:
        chart_data = output

    line_fig = px.line(
        chart_data,
        x="Record ID",
        y="Attack Probability (%)"
    )
    line_fig.update_traces(
        line_color="#00c8ff",
        line_width=2,
        fill="tozeroy",
        fillcolor="rgba(0,200,255,0.12)"
    )
    line_fig.update_layout(
        height=400,
        paper_bgcolor="#102945",
        plot_bgcolor="#102945",
        font=dict(color="white"),
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(line_fig, use_container_width=True)


def show_results(output):
    total = len(output)
    attacks = int((output["Status"] == "Attack").sum())
    normal = total - attacks
    high = int((output["Risk Level"] == "High").sum())

    st.header("Detection Results")

    a, b, c, d = st.columns(4)

    with a:
        metric_card("Total Records", f"{total:,}", "Uploaded records")
    with b:
        metric_card("Detected Attacks", f"{attacks:,}", "Predicted attack records", "#ff5364")
    with c:
        metric_card("High Risk Records", f"{high:,}", "High probability records", "#ffb347")
    with d:
        metric_card("Normal Traffic", f"{normal:,}", "Predicted normal records", "#20c997")

    show_analytics(output)

    st.header("🔴 High Risk Records")
    high_df = output[output["Risk Level"] == "High"]

    if high_df.empty:
        st.info("No high-risk records detected at the current probability threshold.")
    else:
        st.dataframe(high_df, use_container_width=True, height=350, hide_index=True)

    st.header("📋 Complete Detection Results")
    st.dataframe(output, use_container_width=True, height=450, hide_index=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("🛡️ AUTOMATED NIDS")
    st.caption("Cyber Security System")

    page = st.radio(
        "Navigation",
        ["Dashboard", "NIDS Detection", "App Search", "Network Monitor", "Reports"]
    )

    st.divider()
    if os.path.exists(MODEL_PATH):
        st.success("Model Available")
    else:
        st.error("Model Not Found")


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":
    st.title("🛡️ Cyber Security Operations Center")
    st.subheader("Automated Cyber Threat and Network Intrusion Detection")

    if st.session_state.results is None:
        st.info("Upload a CSV in the NIDS Detection page to view dashboard analytics.")
    else:
        show_results(st.session_state.results["output"])


# =====================================================
# NIDS DETECTION
# =====================================================

elif page == "NIDS Detection":
    st.title("🔍 Network Intrusion Detection")
    st.write("Upload a network traffic CSV file to analyze.")

    file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"],
        key="nids_csv_uploader"
    )

    if file is not None:
        try:
            raw_bytes = file.getvalue()
            upload_key = hashlib.md5(raw_bytes).hexdigest()

            # Clear old results when a different file is selected
            if st.session_state.active_upload_key != upload_key:
                st.session_state.active_upload_key = upload_key
                st.session_state.results = None
                st.session_state.report = None
                st.session_state.results_upload_key = None

            df = pd.read_csv(file, low_memory=False)
            df.columns = df.columns.astype(str).str.strip()
            df = df.loc[:, ~df.columns.duplicated()]

            st.success(f"CSV uploaded successfully: {file.name}")

            st.header("📂 Uploaded Dataset Information")
            a, b, c = st.columns(3)

            with a:
                metric_card("Total Records", f"{len(df):,}", "Dataset rows")
            with b:
                metric_card("Total Columns", f"{len(df.columns):,}", "Dataset columns")
            with c:
                metric_card("File Size", f"{file.size / 1024:.2f} KB", "Uploaded file size")

            with st.expander("Preview Dataset"):
                st.dataframe(df.head(20), use_container_width=True, hide_index=True)

            if st.button(
                "🚀 Start Threat Detection",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Analyzing network traffic..."):
                    detection = predict_dataset(df)

                if detection is not None:
                    st.session_state.results = detection
                    st.session_state.report = detection["report"]
                    st.session_state.results_upload_key = upload_key
                    st.success("Detection completed successfully!")

            # Display results only for the currently uploaded file
            if (
                st.session_state.results is not None
                and st.session_state.results_upload_key == upload_key
            ):
                result = st.session_state.results

                if result["missing"]:
                    match_count = max(0, len(result["output"].columns) - len(result["missing"]))
                    st.warning(
                        f"Missing model features: {len(result['missing'])}. "
                        "These features were filled with 0; this may affect prediction."
                    )
                    with st.expander("View missing feature names"):
                        st.write(result["missing"])

                show_results(result["output"])

                report_csv = result["report"].to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ Download Complete Detection Report",
                    report_csv,
                    "NIDS_Detection_Report.csv",
                    "text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"CSV error: {e}")


# =====================================================
# APP SEARCH
# =====================================================

elif page == "App Search":
    st.title("📱 Google Play App Search")
    query = st.text_input("Search app", placeholder="Meesho, Instagram, WhatsApp")

    if st.button("🔍 Search App", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter an app name.")
        elif play_search is None or play_app is None:
            st.error("Google Play Scraper is not installed.")
        else:
            try:
                ids = []
                q = query.lower()

                if "whatsapp" in q:
                    ids = ["com.whatsapp", "com.whatsapp.w4b"]
                elif "instagram" in q:
                    ids = ["com.instagram.android", "com.instagram.lite"]

                search_results = play_search(query, lang="en", country="in", n_hits=15)

                for item in search_results:
                    app_id = item.get("appId")
                    if app_id and app_id not in ids:
                        ids.append(app_id)

                for app_id in ids:
                    try:
                        details = play_app(app_id, lang="en", country="in")
                        st.subheader(details.get("title", app_id))
                        st.write("Developer:", details.get("developer", "N/A"))
                        st.write("Category:", details.get("genre", "N/A"))
                        st.write("Package ID:", app_id)

                        rating = details.get("score")
                        st.metric(
                            "Rating",
                            f"{rating:.1f}/5" if isinstance(rating, (float, int)) else "N/A"
                        )

                        st.write("Installs:", details.get("installs", "N/A"))
                        st.write("Reviews:", details.get("reviews", "N/A"))

                        url = details.get("url") or (
                            "https://play.google.com/store/apps/details?id=" + app_id
                        )
                        st.markdown(f"[Open Google Play listing]({url})")
                        st.divider()

                    except Exception:
                        st.markdown(
                            f"[Open Play Store app]("
                            f"https://play.google.com/store/apps/details?id={app_id})"
                        )

            except Exception as e:
                st.error(f"Search error: {e}")


# =====================================================
# NETWORK MONITOR
# =====================================================

elif page == "Network Monitor":
    st.title("🌐 Network Monitor")
    st.warning("This displays network usage of the computer running this app.")

    if st.button("🔄 Refresh Network", type="primary"):
        counters = psutil.net_io_counters()
        st.session_state.network_history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Sent MB": counters.bytes_sent / (1024 * 1024),
            "Received MB": counters.bytes_recv / (1024 * 1024)
        })
        st.session_state.network_history = st.session_state.network_history[-30:]

    counters = psutil.net_io_counters()
    sent = counters.bytes_sent / (1024 * 1024)
    received = counters.bytes_recv / (1024 * 1024)

    a, b = st.columns(2)
    with a:
        metric_card("Data Sent", f"{sent:.2f} MB", "Since system boot", "#00c8ff")
    with b:
        metric_card("Data Received", f"{received:.2f} MB", "Since system boot", "#20c997")

    st.write("Packets Sent:", f"{counters.packets_sent:,}")
    st.write("Packets Received:", f"{counters.packets_recv:,}")
    st.write("Errors In:", counters.errin)
    st.write("Errors Out:", counters.errout)
    st.write("Dropped In:", counters.dropin)
    st.write("Dropped Out:", counters.dropout)

    if st.session_state.network_history:
        hist = pd.DataFrame(st.session_state.network_history)
        fig = px.line(hist, x="Time", y=["Sent MB", "Received MB"], markers=True)
        fig.update_layout(
            height=400,
            paper_bgcolor="#102945",
            plot_bgcolor="#102945",
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# REPORTS
# =====================================================

elif page == "Reports":
    st.title("📄 NIDS Reports")

    if st.session_state.report is None:
        st.info("Run a detection from the NIDS Detection page first.")
    else:
        output = st.session_state.results["output"]
        report = st.session_state.report.copy()

        total = len(output)
        attacks = int((output["Status"] == "Attack").sum())
        normal = total - attacks
        high = int((output["Risk Level"] == "High").sum())

        a, b, c, d = st.columns(4)
        a.metric("Total Records", f"{total:,}")
        b.metric("Attacks", f"{attacks:,}")
        c.metric("Normal", f"{normal:,}")
        d.metric("High Risk", f"{high:,}")

        st.subheader("Complete Report")
        report = report.loc[:, ~report.columns.duplicated()].copy()
        st.dataframe(report, use_container_width=True, height=500, hide_index=True)

        report_csv = report.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download NIDS Report CSV",
            report_csv,
            "NIDS_Complete_Report.csv",
            "text/csv",
            use_container_width=True
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()
st.markdown(
    "<div style='text-align:center;color:#9bb8d4;padding:15px'>"
    "🛡️ Automated Cyber Threat and Network Intrusion Detection System"
    "<br>Machine Learning Based Network Security Dashboard"
    "</div>",
    unsafe_allow_html=True
)

# ================= MULTI-FORMAT SECURITY FILE ANALYZER =================
import re
import io
import json
import ipaddress
import pandas as pd
import streamlit as st

def extract_file_text(uploaded_file):
    ext = uploaded_file.name.lower().split(".")[-1]
    raw = uploaded_file.getvalue()

    if ext in ["txt", "log", "csv", "json", "html"]:
        text = raw.decode("utf-8", errors="ignore")
        if ext == "html":
            text = re.sub(r"<[^>]+>", " ", text)
        return text

    elif ext == "xml":
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw)
        return " ".join(root.itertext())

    elif ext == "pdf":
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    elif ext == "docx":
        from docx import Document
        doc = Document(io.BytesIO(raw))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)

    elif ext == "xlsx":
        book = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        parts = []
        for sheet_name, df in book.items():
            parts.append(f"Sheet: {sheet_name}")
            parts.append(df.to_string(index=False))
        return "\n".join(parts)

    elif ext in ["pcap", "pcapng"]:
        from scapy.all import rdpcap, IP, TCP, UDP
        packets = rdpcap(io.BytesIO(raw))
        parts = []
        for pkt in packets:
            if IP in pkt:
                src = pkt[IP].src
                dst = pkt[IP].dst
                proto = pkt[IP].proto
                ports = ""
                if TCP in pkt:
                    ports = f" TCP {pkt[TCP].sport}->{pkt[TCP].dport}"
                elif UDP in pkt:
                    ports = f" UDP {pkt[UDP].sport}->{pkt[UDP].dport}"
                parts.append(f"{src} -> {dst} Protocol:{proto}{ports}")
        return "\n".join(parts)

    else:
        raise ValueError("Unsupported file format")

def analyze_security_text(text):
    keywords = {
        "failed login": "Failed Login Attempts",
        "login failed": "Failed Login Attempts",
        "unauthorized": "Unauthorized Access",
        "brute force": "Brute Force",
        "malware": "Malware Indicator",
        "ransomware": "Ransomware Indicator",
        "sql injection": "SQL Injection Indicator",
        "xss": "XSS Indicator",
        "denial of service": "DoS Indicator",
        "ddos": "DDoS Indicator",
        "port scan": "Port Scan Indicator",
        "suspicious": "Suspicious Activity",
        "attack": "Attack Keyword",
        "blocked": "Blocked Event",
    }

    found = []
    lower_text = text.lower()

    for keyword, label in keywords.items():
        count = lower_text.count(keyword)
        if count:
            found.append((label, keyword, count))

    # Find IP addresses in extracted text
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ips = set(re.findall(ip_pattern, text))
    valid_ips = []

    for ip in ips:
        try:
            ipaddress.ip_address(ip)
            valid_ips.append(ip)
        except ValueError:
            pass

    risk_score = min(100, len(found) * 15 + len(valid_ips) * 3)

    if risk_score >= 60:
        risk_level = "High"
    elif risk_score >= 25:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return found, sorted(valid_ips), risk_score, risk_level

# Analyzer UI
st.markdown("---")
st.header("📂 Multi-Format Security File Analyzer")
st.caption("Upload security-related files for keyword and IP indicator analysis.")

uploaded_files = st.file_uploader(
    "Upload files",
    type=["csv", "pdf", "docx", "txt", "log", "json",
          "xml", "html", "xlsx", "pcap", "pcapng"],
    accept_multiple_files=True,
    key="multi_security_upload"
)

if uploaded_files:
    if st.button("🔍 Analyze Uploaded Files", key="multi_format_analyze"):
        for file in uploaded_files:
            st.markdown(f"### 📄 {file.name}")

            try:
                extracted_text = extract_file_text(file)

                if not extracted_text.strip():
                    st.warning("No readable text found in this file.")
                    continue

                findings, ips, score, risk = analyze_security_text(extracted_text)

                st.metric("Risk Level", risk, help="Heuristic score, not ML prediction")
                st.metric("Risk Score", f"{score}/100")

                st.write("**Suspicious Indicators**")
                if findings:
                    st.dataframe(
                        pd.DataFrame(findings,
                            columns=["Indicator", "Keyword", "Occurrences"]),
                        use_container_width=True
                    )
                else:
                    st.info("No suspicious keywords matched.")

                st.write("**IP Addresses Found**")
                if ips:
                    st.write(", ".join(ips))
                else:
                    st.write("No IP addresses found.")

                report = (
                    f"File: {file.name}\n"
                    f"Risk Level: {risk}\n"
                    f"Risk Score: {score}/100\n"
                    f"Indicators: {findings}\n"
                    f"IPs: {ips}\n"
                )

                st.download_button(
                    "📥 Download Analysis Report",
                    data=report,
                    file_name=f"{file.name}_security_report.txt",
                    mime="text/plain",
                    key=f"download_{file.name}"
                )

            except Exception as e:
                st.error(f"Could not analyze {file.name}: {e}")