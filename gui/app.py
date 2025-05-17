import streamlit as st
import requests
import tempfile
import cv2
import os
import re
import subprocess
import datetime
import login
import subprocess
import json
import pytz
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import altair as alt

# Backend URL
server_url = "http://127.0.0.1:8000/predict/"
st.set_page_config(layout="wide", page_title="Live Monitoring Dashboard")
# Define the timezone
pakistan_tz = pytz.timezone("Asia/Karachi")

# Get the current time in Pakistan timezone
current_time = datetime.datetime.now(pakistan_tz).strftime("%Y-%m-%d %H:%M:%S")

token = st.session_state.get("access_token")  # Get the JWT token
# Initialize session state variables once to keep them across tabs
if "anomaly_scores" not in st.session_state:
    st.session_state.anomaly_scores = []

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "detection" not in st.session_state:
    st.session_state.detection = False

if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None

def get_video_rotation(video_path):
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate", "-of", "json", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = json.loads(result.stdout)
        tags = output.get("streams", [{}])[0].get("tags", {})
        return int(tags.get("rotate", 0))
    except Exception as e:
        return 0  # default if rotation info not found

# Check if the user is logged in
if 'user_logged_in' not in st.session_state or not st.session_state['user_logged_in']:
    login.user_login()  # Show the login page
else:
    # Professional Light Theme with Elegant Contrast
    st.markdown("""
        <style>
        /* Overall background and text */
        html, body, [class*="css"]  {
            background-color: #e9eff5 !important;
            color: #c78787 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
    
        /* Container styling */
        .block-container {
            padding: 2rem 3rem;
            background-color: #e9eff5;
        }
    
        /* Title and headings */
        h1, h2, h3 {
            color: #8383d4;
        }
    
        /* Sidebar */
        .css-6qob1r {
            background-color: #e9eff5;
            box-shadow: 2px 0 5px rgba(0,0,0,0.05);
        }
    
        /* Buttons */
        .stButton>button, .stDownloadButton>button {
            background-color: #0056b3;
            color: white;
            font-weight: 600;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: background-color 0.3s ease;
        }
    
        .stButton>button:hover, .stDownloadButton>button:hover {
            background-color: #004494;
        }
    
        /* File uploader, inputs, select boxes */
        .stTextInput, .stFileUploader, .stSelectbox {
            background-color: #3c3c7d !important;
            border: 1px solid #ccc !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            padding: 8px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
    
        /* Metrics cards */
        .stMetric {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        }
    
        /* Alerts container */
        .stMarkdown, .element-container {
            margin-bottom: 1rem;
            
        }
    
        .stAlert {
            background-color: #e9eff5;
            border-left: 5px solid #d9534f;
            padding: 10px;
            font-weight: bold;
            color: #b20000;
            border-radius: 6px;
        }
    
        /* Success messages */
        .stSuccess {
            background-color: #e6ffed;
            border-left: 5px solid #28a745;
            padding: 10px;
            font-weight: bold;
            color: #155724;
            border-radius: 6px;
        }
    
        /* Tabs and option menu styling */
        .css-1d391kg {
            background: #e9eff5;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    with st.sidebar:
        selected = option_menu(
            "Menu", ["Live Feed", "Analytics", "Alerts", "Settings"],
            icons=['camera-video', 'bar-chart', 'bell', 'gear'],
            menu_icon="cast", default_index=0
        )
    
    # Topbar
    col1, col2, col3 = st.columns([6, 2, 1])
    with col1:
        st.markdown("### 💻 Transformer-Based Anomaly Detection")
    with col2:
        st.success("🟢 Detection Active  |  Cameras Online")
    with col3:
        st.markdown(f"⏰ **{current_time}**")
    st.markdown("---")

    # Real-time Alerts: Display the alerts dynamically
    # Main Layout
    if selected == "Live Feed":
        col1, col2 = st.columns([3, 1])
    
        with col1:
            st.subheader("🎥 Live Video Feed")
            uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
            start_col, stop_col, log_col = st.columns(3)
    
            if start_col.button("▶️ Start Detection", use_container_width=True):
                if uploaded_file is not None:
                    st.session_state['detection'] = True
    
                    # Save uploaded video temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(uploaded_file.read())
                        video_path = tmp.name
                    
                    # Save the video path in session state for later use
                    st.session_state.uploaded_video_path = video_path

                    st.info("Sending video to backend to get anomaly scores...")
            
                    try:
                        # Get token from session state
                        token = st.session_state.get("access_token")
                        # Send video to FastAPI backend WITH AUTH HEADER
                        headers = {"Authorization": f"Bearer {token}"}
                        with open(video_path, "rb") as f:
                            files = {"file": ("video.mp4", f, "video/mp4")}
                            response = requests.post(server_url, files=files, headers=headers, stream=True)
            
                        if response.status_code != 200:
                            st.error(f"Backend Error {response.status_code}: {response.text}")
                        else:
                            scores = []
                            for line in response.iter_lines():
                                if line:
                                    decoded = line.decode("utf-8")
                                    if decoded.startswith("data:"):
                                        raw = decoded.replace("data:", "").strip()
                                        match = re.search(r"\[\[(.*?)\]\]", raw)
                                        if match:
                                            score = float(match.group(1))
                                            scores.append(score)
                            st.session_state.anomaly_scores = scores
            
                            st.success("Received scores. Annotating video...")

                            cap = cv2.VideoCapture(video_path)
                            fps = int(cap.get(cv2.CAP_PROP_FPS))
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
                            annotated_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))
            
                            segment_idx = 0
                            frame_count = 0
                            segment_length = 16
            
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                    
                                frame = cv2.flip(frame, -1)
                                
                                if segment_idx < len(scores):
                                    score = scores[segment_idx]
                                else:
                                    score = 0.0
                                # Update session state with real-time alerts
                                if score > 0.5:  # Customize this threshold as needed
                                    alert_msg = f"🚨 Anomaly detected at frame {frame_count + 1}!"
                                    st.session_state.alerts.append(alert_msg)  # Append the alert to session state
            
                                cv2.putText(frame, f"Anomaly Score: {score:.2f}", (30, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                                out.write(frame)
                                frame_count += 1
                                if frame_count % segment_length == 0:
                                    segment_idx += 1
            
                            cap.release()
                            out.release()
            
                            final_video_path = os.path.join(tempfile.gettempdir(), "final_annotated_output.mp4")
                            ffmpeg_path = "/home/jovyan/ffmpeg-7.0.2-amd64-static/ffmpeg"
                            ffmpeg_command = [
                                ffmpeg_path, "-y",
                                "-i", annotated_path,
                                "-vcodec", "libx264",
                                "-acodec", "aac",
                                final_video_path
                            ]
                            subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
                            st.success("✅ Annotated video is ready!")
                            with open(final_video_path, "rb") as video_file:
                                st.video(video_file.read())
            
                            with open(final_video_path, "rb") as f:
                                st.download_button("📥 Download Annotated Video", f, "annotated_video.mp4")
            
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
            else:
                st.warning("⚠️ Please upload a video before starting detection.")
    
            if stop_col.button("⏹ Stop Detection", use_container_width=True):
                st.session_state['detection'] = False
            if log_col.button("📜 View Logs", use_container_width=True):
                st.info("Logs will be displayed here...")
    
            st.image("https://via.placeholder.com/600x300?text=Live+Feed+Frame", caption="Live Feed Frame")
    
        with col2:
            st.subheader("🚨 Real-time Alerts")
            
            # Dynamically display real-time alerts stored in session state
            if st.session_state.alerts:
                for alert in st.session_state.alerts:
                    st.markdown(f"- {alert}")
            else:
                st.markdown("No alerts yet. The system is monitoring video frames...")
                
    elif selected == "Analytics":
        st.subheader("📊 Analytics")

        if 'anomaly_scores' in st.session_state and st.session_state.anomaly_scores:
    
            scores = st.session_state.anomaly_scores
    
            # Build DataFrame
            df = pd.DataFrame({
                "Segment": list(range(1, len(scores)+1)),
                "Anomaly Score": scores
            })
    
            # Create line chart
            chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X('Segment', title='Video Segment'),
                y=alt.Y('Anomaly Score', scale=alt.Scale(domain=[0, 1])),
                tooltip=['Segment', 'Anomaly Score']
            ).properties(
                title="Anomaly Score Over Time",
                width=700,
                height=400
            )
    
            st.altair_chart(chart, use_container_width=True)
    
            # Show stats
            st.write("### 📈 Summary Stats")
            st.metric("Max Anomaly Score", f"{max(scores):.2f}")
            st.metric("Average Anomaly Score", f"{sum(scores)/len(scores):.2f}")
            st.metric("Anomalous Segments (score > 0.5)", f"{sum(s > 0.5 for s in scores)} / {len(scores)}")
    
            # Optional: Download CSV
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Anomaly Scores", csv, "anomaly_scores.csv", "text/csv")
    
        else:
            st.info("No scores available yet. Please upload and process a video.")

    
    elif selected == "Alerts":
        st.subheader("📁 Alert History")
    
        # Initialize the alerts list if it doesn't exist
        if 'alerts' not in st.session_state:
            st.session_state['alerts'] = []
    
        if st.session_state['alerts']:
            for alert in reversed(st.session_state['alerts']):
                st.markdown(f"- {alert}")
        else:
            st.info("No alerts recorded yet. Real-time alerts will appear here after detection.")

    
    elif selected == "Settings":
        st.subheader("⚙️ Settings")
        theme = st.selectbox("Choose Theme", ["Dark", "Light"])
        st.success(f"Theme set to: {theme}")
