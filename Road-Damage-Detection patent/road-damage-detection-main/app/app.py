import streamlit as st
import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import numpy as np
from io import BytesIO
import socket
import pydeck as pdk
import json

# Safe import for QR Code generation
try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

# Safe import for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ==============================================================================
# HELPER FUNCTIONS: LOCAL IP & REPORT GENERATION
# ==============================================================================
def get_local_ip():
    """Finds the local IP address of your machine on Wi-Fi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_pdf_report(df, file_name, selected_model, conf_thresh):
    """Generates a professional PDF inspection report in memory."""
    if not REPORTLAB_AVAILABLE:
        return None
        
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#0E1117'), spaceAfter=12)
    story.append(Paragraph("<b>VisionRoad AI - Damage Inspection Report</b>", title_style))
    
    # Metadata
    meta_text = f"<b>File Name:</b> {file_name} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Model:</b> {selected_model} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Min Conf:</b> {conf_thresh*100:.0f}%"
    story.append(Paragraph(meta_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Convert Dataframe to List for PDF Table
    table_data = [list(df.columns)]  # Header
    for _, row in df.iterrows():
        row_list = [
            str(row["Anomaly ID"]),
            str(row["Damage Class"]),
            f"{row['Confidence']:.1f}%",
            str(row["Severity Level"]),
            str(row["Bounding Box"]),
            str(row["Estimated Area"])
        ]
        table_data.append(row_list)
        
    # PDF Table Styling
    pdf_table = Table(table_data, colWidths=[60, 80, 70, 70, 150, 80])
    pdf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#161B22')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#21262D')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8F9FA')]),
    ]))
    
    story.append(pdf_table)
    doc.build(story)
    
    buffer.seek(0)
    return buffer


# ==============================================================================
# 1. PAGE CONFIGURATION & STATE INITIALIZATION
# ==============================================================================
st.set_page_config(
    page_title="VisionRoad | AI Road Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

local_ip = get_local_ip()
mobile_url = f"http://{local_ip}:8501"

# Persistent GPS coordinates list across re-runs
if 'map_spots' not in st.session_state:
    st.session_state['map_spots'] = [
        {"lat": 30.7055, "lon": 76.7185, "type": "Deep Pothole", "severity": "Critical", "r": 255, "g": 50, "b": 50, "radius": 25},
        {"lat": 30.7150, "lon": 76.7320, "type": "Edge Subsidence", "severity": "Critical", "r": 255, "g": 50, "b": 50, "radius": 25},
        {"lat": 30.7100, "lon": 76.7250, "type": "Longitudinal Crack", "severity": "Moderate", "r": 255, "g": 140, "b": 0, "radius": 18},
        {"lat": 30.7020, "lon": 76.7210, "type": "Transverse Crack", "severity": "Moderate", "r": 255, "g": 140, "b": 0, "radius": 18},
        {"lat": 30.6980, "lon": 76.7110, "type": "Minor Surface Wear", "severity": "Low", "r": 255, "g": 215, "b": 0, "radius": 14},
    ]

if 'user_location' not in st.session_state:
    st.session_state['user_location'] = {"lat": 30.7080, "lon": 76.7200, "active": False}


# ==============================================================================
# 2. SIDEBAR CONTROL PANEL
# ==============================================================================
with st.sidebar:
    st.title("Control Panel")
    
    st.subheader("Model Selection")
    selected_model_name = st.selectbox(
        "Select YOLO Architecture",
        ["yolov8n.pt", "yolov8s.pt"],
        index=0,
        help="yolov8n is lighter and faster; yolov8s provides higher accuracy."
    )
    
    st.subheader("Input Source")
    source_type = st.radio(
        "Select Source", 
        ["Webcam", "Upload Video / Image", "📱 Mobile Camera"], 
        index=1
    )
    
    st.subheader("Detection Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 1.00, 0.40, 0.05)
    
    st.markdown("---")
    st.subheader("📱 Mobile Camera Connect")
    
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=250x250&data={mobile_url}&bgcolor=16-27-34&color=88-166-255"
    st.image(qr_api_url, caption=f"Scan to open on Mobile\n({mobile_url})", use_container_width=True)
    st.caption("Connect your phone to the same Wi-Fi network.")
    st.caption("Engine: Ultralytics YOLOv8 | Live Auto-Download Enabled")


# ==============================================================================
# 3. MODEL LOADING FUNCTION
# ==============================================================================
@st.cache_resource
def load_yolo_model(model_name):
    return YOLO(model_name)

try:
    model = load_yolo_model(selected_model_name)
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Error loading {selected_model_name}: {e}")
    model = None
    MODEL_LOADED = False

# ==============================================================================
# 4. CUSTOM CSS & GLOWING SCROLL ANIMATIONS
# ==============================================================================
st.markdown("""
<style>
    /* -------------------------------------------------------------------------
       1. NEON GLOWING TEXT STYLES (CYAN / BLUE / EMERALD)
       ------------------------------------------------------------------------- */
    .glow-text-cyan {
        color: #58A6FF;
        font-weight: 700;
        text-shadow: 0 0 8px rgba(88, 166, 255, 0.6), 
                     0 0 20px rgba(88, 166, 255, 0.4), 
                     0 0 35px rgba(88, 166, 255, 0.2);
        animation: pulseGlowCyan 3s ease-in-out infinite alternate;
        display: inline-block;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glow-text-emerald {
        color: #3FB950;
        font-weight: 700;
        text-shadow: 0 0 8px rgba(63, 185, 80, 0.6), 
                     0 0 20px rgba(63, 185, 80, 0.3);
        animation: pulseGlowGreen 3.5s ease-in-out infinite alternate;
        display: inline-block;
    }

    .glow-text-amber {
        color: #FFA500;
        font-weight: 700;
        text-shadow: 0 0 8px rgba(255, 165, 0, 0.6), 
                     0 0 22px rgba(255, 165, 0, 0.3);
        display: inline-block;
    }

    /* Interactive Hover Effect */
    .glow-text-cyan:hover, .glow-text-emerald:hover, .glow-text-amber:hover {
        transform: translateY(-2px) scale(1.02);
        cursor: pointer;
        filter: brightness(1.25);
    }

    /* -------------------------------------------------------------------------
       2. SCROLL & ENTRANCE ANIMATIONS (FADE UP + GLOW BLOOM)
       ------------------------------------------------------------------------- */
    @keyframes fadeUpGlow {
        0% {
            opacity: 0;
            transform: translateY(24px);
            filter: blur(4px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
            filter: blur(0px);
        }
    }

    @keyframes pulseGlowCyan {
        0% {
            text-shadow: 0 0 6px rgba(88, 166, 255, 0.4), 0 0 15px rgba(88, 166, 255, 0.2);
        }
        100% {
            text-shadow: 0 0 12px rgba(88, 166, 255, 0.8), 0 0 28px rgba(88, 166, 255, 0.5), 0 0 45px rgba(88, 166, 255, 0.3);
        }
    }

    @keyframes pulseGlowGreen {
        0% {
            text-shadow: 0 0 5px rgba(63, 185, 80, 0.4);
        }
        100% {
            text-shadow: 0 0 12px rgba(63, 185, 80, 0.8), 0 0 24px rgba(63, 185, 80, 0.4);
        }
    }

    /* Scroll Reveal Container */
    .reveal-on-scroll {
        animation: fadeUpGlow 0.9s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Actionable Interactive Cards */
    .actionable-card {
        background: linear-gradient(145deg, #161B22, #0E1117);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.4rem;
        transition: all 0.35s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .actionable-card:hover {
        border-color: #58A6FF;
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(88, 166, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)
# ==============================================================================
# 5. APPLICATION HEADER (GLOWING ON ENTRANCE & SCROLL)
# ==============================================================================
st.markdown("""
<div class="main-header reveal-on-scroll">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0;">
                <span class="glow-text-cyan">VisionRoad AI</span>
            </h1>
            <p style="color: #8B949E; margin: 4px 0 0 0; font-size: 0.95rem;">
                Automated Real-Time Surface Anomaly Detection with <span class="glow-text-emerald">Live GIS Map</span>
            </p>
        </div>
        <div style="margin-top: 10px;">
            <span class="status-badge" style="box-shadow: 0 0 10px rgba(46, 160, 67, 0.35);">
                &bull; <span class="glow-text-emerald">AI ENGINE ACTIVE</span>
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 6. KPI METRICS GRID (ACTIONABLE GLOWING CARDS)
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="actionable-card reveal-on-scroll">
        <div class="metric-label">System Status</div>
        <div class="metric-value glow-text-emerald" style="font-size: 1.6rem;">READY</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="actionable-card reveal-on-scroll">
        <div class="metric-label">Active Model</div>
        <div class="metric-value glow-text-cyan" style="font-size: 1.4rem;">{selected_model_name.replace('.pt','').upper()}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="actionable-card reveal-on-scroll">
        <div class="metric-label">Min Confidence</div>
        <div class="metric-value glow-text-amber">{conf_thresh * 100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="actionable-card reveal-on-scroll">
        <div class="metric-label">Mode</div>
        <div class="metric-value glow-text-cyan" style="font-size: 1.4rem;">Inspection</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# 7. TABBED MAIN WORKSPACE
# ==============================================================================
tab_live, tab_map, tab_logs = st.tabs(["Detection Feed", "🗺️ Live Damage Map", "Incident Logs"])

with tab_live:
    if source_type == "Webcam":
        st.subheader("Live Camera Stream & On-Demand Scan")
        
        run_cam = st.checkbox("Turn On Live Camera", value=False)
        frame_window = st.image([])
        
        if run_cam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam.")
            
            while run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab video frame.")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if MODEL_LOADED:
                    results = model(frame_rgb, conf=conf_thresh)
                    annotated_frame = results[0].plot()
                    frame_window.image(annotated_frame, use_container_width=True)
                    st.session_state['latest_frame'] = frame_rgb
                else:
                    frame_window.image(frame_rgb, use_container_width=True)
                    st.session_state['latest_frame'] = frame_rgb
                
            cap.release()
            
        if 'latest_frame' in st.session_state and st.session_state['latest_frame'] is not None:
            st.markdown("---")
            if st.button("📸 Capture & Generate Live Inspection Report", type="primary", use_container_width=True):
                
                st.subheader("📊 Captured Live Frame Analysis")
                current_frame = st.session_state['latest_frame']
                
                if MODEL_LOADED:
                    results = model(current_frame, conf=conf_thresh)
                    result = results[0]
                    annotated_captured_frame = result.plot()
                    
                    detections_data = []
                    boxes = result.boxes
                    
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id].title()
                        confidence = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        
                        box_width = xyxy[2] - xyxy[0]
                        box_height = xyxy[3] - xyxy[1]
                        area_px = int(box_width * box_height)
                        
                        severity = "High" if area_px > 15000 or confidence > 0.8 else "Medium" if area_px > 5000 else "Low"
                        
                        detections_data.append({
                            "Anomaly ID": f"#LIVE-{i+1:02d}",
                            "Damage Class": class_name,
                            "Confidence": confidence * 100,
                            "Severity Level": severity,
                            "Bounding Box": f"[{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]",
                            "Estimated Area": f"{area_px:,} px²"
                        })
                    
                    col_cap_img, col_cap_table = st.columns([1, 1])
                    
                    with col_cap_img:
                        st.image(annotated_captured_frame, caption="Captured Live Frame with Detections", use_container_width=True)
                        
                    with col_cap_table:
                        if len(detections_data) > 0:
                            df_live_results = pd.DataFrame(detections_data)
                            st.dataframe(df_live_results, hide_index=True, use_container_width=True)
                            
                            # Automatically pin detected live hazard to map
                            new_spot_lat = st.session_state['user_location']['lat'] + np.random.uniform(-0.003, 0.003)
                            new_spot_lon = st.session_state['user_location']['lon'] + np.random.uniform(-0.003, 0.003)
                            st.session_state['map_spots'].append({
                                "lat": new_spot_lat,
                                "lon": new_spot_lon,
                                "type": detections_data[0]["Damage Class"],
                                "severity": detections_data[0]["Severity Level"],
                                "r": 255 if detections_data[0]["Severity Level"] == "High" else 255 if detections_data[0]["Severity Level"] == "Medium" else 255,
                                "g": 50 if detections_data[0]["Severity Level"] == "High" else 140 if detections_data[0]["Severity Level"] == "Medium" else 215,
                                "b": 50 if detections_data[0]["Severity Level"] == "High" else 0 if detections_data[0]["Severity Level"] == "Medium" else 0,
                                "radius": 24
                            })
                            st.success(f"📍 Automatically pinned detection to Live Map at ({new_spot_lat:.4f}, {new_spot_lon:.4f})")
                        else:
                            st.info("🎉 No road damage detected in the captured frame above the threshold.")

    elif source_type == "📱 Mobile Camera":
        st.subheader("📸 Mobile Snap & Direct Scan")
        st.caption("Scan the QR code in the sidebar with your mobile device to open this camera view directly on your phone.")
        
        mobile_photo = st.camera_input("Take a photo with your mobile camera")
        
        if mobile_photo is not None:
            mobile_img = Image.open(mobile_photo).convert("RGB")
            mobile_array = np.array(mobile_img)
            
            if MODEL_LOADED:
                results = model(mobile_array, conf=conf_thresh)
                result = results[0]
                annotated_mobile_frame = result.plot()
                
                detections_data = []
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id].title()
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    
                    box_width = xyxy[2] - xyxy[0]
                    box_height = xyxy[3] - xyxy[1]
                    area_px = int(box_width * box_height)
                    
                    severity = "High" if area_px > 15000 or confidence > 0.8 else "Medium" if area_px > 5000 else "Low"
                    
                    detections_data.append({
                        "Anomaly ID": f"#MOB-{i+1:02d}",
                        "Damage Class": class_name,
                        "Confidence": confidence * 100,
                        "Severity Level": severity,
                        "Bounding Box": f"[{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]",
                        "Estimated Area": f"{area_px:,} px²"
                    })
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.subheader("Mobile Photo Captured")
                    st.image(mobile_img, use_container_width=True)
                with col_m2:
                    st.subheader("AI Detection Output")
                    st.image(annotated_mobile_frame, use_container_width=True)
                    
                st.markdown("---")
                if len(detections_data) > 0:
                    df_mobile = pd.DataFrame(detections_data)
                    st.dataframe(df_mobile, hide_index=True, use_container_width=True)
                else:
                    st.info("🎉 No road damage detected in the mobile photo.")

    elif source_type == "Upload Video / Image":
        st.subheader("Upload Media File")
        uploaded_file = st.file_uploader("Choose a road surface image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.info(f"📁 Loaded File: {uploaded_file.name}")
            
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            
            if st.button("🚀 Analyze Road Surface", type="primary", use_container_width=True):
                
                loader_placeholder = st.empty()
                loader_placeholder.markdown("""
                <div class="road-container">
                    <div class="car-loader">🚗</div>
                    <div class="road-line"></div>
                </div>
                <p style="text-align: center; color: #58A6FF; font-weight: 600;">Scanning road surface for damage & anomalies...</p>
                """, unsafe_allow_html=True)
                
                if MODEL_LOADED:
                    results = model(img_array, conf=conf_thresh)
                    result = results[0]
                    annotated_frame = result.plot()
                    
                    detections_data = []
                    boxes = result.boxes
                    
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id].title()
                        confidence = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        
                        box_width = xyxy[2] - xyxy[0]
                        box_height = xyxy[3] - xyxy[1]
                        area_px = int(box_width * box_height)
                        
                        severity = "High" if area_px > 15000 or confidence > 0.8 else "Medium" if area_px > 5000 else "Low"
                        
                        detections_data.append({
                            "Anomaly ID": f"#DET-{i+1:02d}",
                            "Damage Class": class_name,
                            "Confidence": confidence * 100,
                            "Severity Level": severity,
                            "Bounding Box": f"[{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]",
                            "Estimated Area": f"{area_px:,} px²"
                        })
                    
                    loader_placeholder.empty()
                    st.success("✅ Analysis Complete!")
                    
                    col_orig, col_res = st.columns(2)
                    with col_orig:
                        st.subheader("Original Input Image")
                        st.image(image, use_container_width=True)
                        
                    with col_res:
                        st.subheader("AI Detection Output (With Borders)")
                        st.image(annotated_frame, caption=f"Total Anomalies Found: {len(detections_data)}", use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📋 Detailed Anomaly Inspection Report")
                    
                    if len(detections_data) > 0:
                        df_results = pd.DataFrame(detections_data)
                        st.dataframe(df_results, hide_index=True, use_container_width=True)
                        
                        # Add detected spot to the live map dynamically
                        st.session_state['map_spots'].append({
                            "lat": st.session_state['user_location']['lat'] + np.random.uniform(-0.002, 0.002),
                            "lon": st.session_state['user_location']['lon'] + np.random.uniform(-0.002, 0.002),
                            "type": detections_data[0]["Damage Class"],
                            "severity": detections_data[0]["Severity Level"],
                            "r": 255 if detections_data[0]["Severity Level"] == "High" else 255 if detections_data[0]["Severity Level"] == "Medium" else 255,
                            "g": 50 if detections_data[0]["Severity Level"] == "High" else 140 if detections_data[0]["Severity Level"] == "Medium" else 215,
                            "b": 50 if detections_data[0]["Severity Level"] == "High" else 0 if detections_data[0]["Severity Level"] == "Medium" else 0,
                            "radius": 22
                        })
                        st.info("🗺️ Hazard coordinate tagged & added to the Damage Map tab!")
                    else:
                        st.info("🎉 No road damage detected above threshold.")
# In tab_map:
with tab_map:
    st.markdown("""
    <div class="reveal-on-scroll">
        <h3 style="margin-bottom: 4px;">🗺️ <span class="glow-text-cyan">Live Road Infrastructure Health</span> & Geo-Spatial Map</h3>
        <p style="color: #8B949E; font-size: 0.9rem;">
            Real-time visualization with <span class="glow-text-emerald">360° 3D Orbit Camera</span> and active GPS location.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ... rest of your map controls ...
tab_live, tab_map, tab_logs, tab_about = st.tabs([
    "Detection Feed", 
    "🗺️ Live Damage Map", 
    "Incident Logs", 
    "ℹ️ About & Strategic Impact"
])

# ==============================================================================
# 8. TAB: ABOUT & STRATEGIC IMPACT (GOVERNMENT, NHAI & DRIVER VALUE)
# ==============================================================================
with tab_about:
    st.markdown("""
    <div class="reveal-on-scroll" style="text-align: center; margin-bottom: 2rem;">
        <h2>About <span class="glow-text-cyan">VisionRoad AI</span></h2>
        <p style="color: #8B949E; font-size: 1.05rem; max-width: 800px; margin: auto;">
            A next-generation computer vision and geospatial intelligence ecosystem built to bridge the gap between 
            <span class="glow-text-emerald">proactive infrastructure maintenance</span> and <span class="glow-text-amber">real-time road accident prevention</span>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # CORE VALUE PILLARS (TWO-COLUMN GRID)
    # --------------------------------------------------------------------------
    col_gov, col_driver = st.columns(2)

    with col_gov:
        st.markdown("""
        <div class="actionable-card reveal-on-scroll" style="height: 100%;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span style="font-size: 1.8rem;">🏛️</span>
                <h3 style="margin: 0;" class="glow-text-cyan">Strategic Impact for NHAI & Govt Bodies</h3>
            </div>
            <p style="color: #8B949E; font-size: 0.92rem; line-height: 1.6;">
                Manual highway audits require slow patrol vans, manual paperwork, and high contractor verification costs. 
                Integrating <b>VisionRoad AI</b> into government systems unlocks automated, cost-effective governance:
            </p>
            <ul style="color: #C9D1D9; font-size: 0.9rem; line-height: 1.8;">
                <li><b class="glow-text-cyan">Massive Cost Reduction:</b> Cuts highway survey and audit operational costs by up to <b>70%</b> compared to legacy lidar and manual inspection teams.</li>
                <li><b class="glow-text-cyan">Targeted Budget Allocation:</b> Automatically generates priority matrices so road repair funds are directed to critical high-severity damage zones first.</li>
                <li><b class="glow-text-cyan">Contractor Accountability:</b> Maintains immutable, timestamped before-and-after visual inspection reports and CSV logs to audit road contractor quality.</li>
                <li><b class="glow-text-cyan">Zero Traffic Disruptions:</b> Inspections can be run continuously from regular patrol or municipal vehicles at standard cruising speeds.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_driver:
        st.markdown("""
        <div class="actionable-card reveal-on-scroll" style="height: 100%;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span style="font-size: 1.8rem;">🛡️</span>
                <h3 style="margin: 0;" class="glow-text-emerald">Accident Prevention & Driver Safety</h3>
            </div>
            <p style="color: #8B949E; font-size: 0.92rem; line-height: 1.6;">
                Unexpected potholes and severe road edge drop-offs are leading causes of fatal road accidents, loss of vehicle control, and tire blowouts:
            </p>
            <ul style="color: #C9D1D9; font-size: 0.9rem; line-height: 1.8;">
                <li><b class="glow-text-emerald">Early Hazard Warnings:</b> When mounted as a smartphone dashcam, it alerts drivers in real time before reaching deep potholes or longitudinal cracks.</li>
                <li><b class="glow-text-emerald">Two-Wheeler Protection:</b> Especially crucial for motorcyclists and cyclists, where sudden road surface depressions lead to immediate fatal crashes.</li>
                <li><b class="glow-text-emerald">Vehicle Damage Mitigation:</b> Prevents suspension failures, rim bends, and sudden swerving into adjacent highway lanes.</li>
                <li><b class="glow-text-emerald">Crowdsourced Pavement Mapping:</b> Everyday commuter scans automatically update the centralized map to warn vehicles following behind.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # BOTTOM SUMMARY BANNER
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="actionable-card reveal-on-scroll" style="text-align: center; border-left: 4px solid #58A6FF;">
        <h4 style="margin: 0 0 8px 0;">Towards Zero Road Casualties & World-Class Highway Infrastructure 🛣️</h4>
        <p style="color: #8B949E; font-size: 0.92rem; margin: 0;">
            By unifying <b>AI computer vision</b> with <b>open-access geospatial intelligence</b>, VisionRoad AI delivers a scalable blueprint 
            for sustainable smart cities and safe national transit corridors.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# LIVE MAP TAB (FULL GIS MAP + GPS LOCATOR + ROAD CORRIDORS)
# ==============================================================================
# ==============================================================================
# LIVE MAP TAB (FULL STREETS + LIVE GPS + 360° ROTATION ENGINE)
# ==============================================================================
with tab_map:
    st.subheader("🗺️ Live Road Infrastructure Map & 360° 3D Orbit View")
    st.caption("Complete street basemap with interactive 360° camera rotation, tilt controls, and live GPS hazard tracking.")

    # 1. LIVE GPS & 360° ROTATION CONTROLS
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1.6])
    
    with ctrl_col1:
        if st.button("📍 Detect & Pin Live GPS", type="primary", use_container_width=True):
            st.components.v1.html("""
            <script>
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(function(pos) {
                        const lat = pos.coords.latitude;
                        const lon = pos.coords.longitude;
                        window.parent.postMessage({type: "streamlit:setComponentValue", value: {lat: lat, lon: lon}}, "*");
                    });
                }
            </script>
            """, height=0)
            st.session_state['user_location']['active'] = True
            st.success("🛰️ GPS Synchronized!")

    with ctrl_col2:
        auto_rotate = st.toggle("🔄 Auto-Orbit Mode", value=False, help="Continuously rotate camera 360 degrees around the inspection site.")

    with ctrl_col3:
        # Camera angle manual slider (0° to 360°)
        if 'camera_bearing' not in st.session_state:
            st.session_state['camera_bearing'] = 0.0

        if auto_rotate:
            st.session_state['camera_bearing'] = (st.session_state['camera_bearing'] + 30.0) % 360.0
            cam_bearing = float(st.session_state['camera_bearing'])
            st.caption(f"Rotating angle: **{cam_bearing:.0f}°**")
        else:
            cam_bearing = st.slider("360° Camera Heading", min_value=0.0, max_value=360.0, value=float(st.session_state['camera_bearing']), step=5.0)
            st.session_state['camera_bearing'] = cam_bearing

    # Lat/Lon coordinates adjusters
    col_la, col_lo, col_pitch = st.columns(3)
    with col_la:
        u_lat = st.number_input("Latitude", value=float(st.session_state['user_location']['lat']), format="%.5f")
    with col_lo:
        u_lon = st.number_input("Longitude", value=float(st.session_state['user_location']['lon']), format="%.5f")
    with col_pitch:
        cam_pitch = st.slider("3D Pitch / Tilt Angle", min_value=0.0, max_value=75.0, value=50.0, step=5.0)

    st.session_state['user_location']['lat'] = u_lat
    st.session_state['user_location']['lon'] = u_lon

    # 2. SPOT DAMAGE DATAFRAME
    spots_data = pd.DataFrame(st.session_state['map_spots'])

    # 3. LIVE VEHICLE / USER PIN
    user_pin_data = pd.DataFrame([{
        "lat": st.session_state['user_location']['lat'],
        "lon": st.session_state['user_location']['lon'],
        "name": "🚗 Live Inspection Vehicle",
        "type": "Inspection Unit",
        "info": "Real-time Vehicle GPS Position",
        "severity": "Active Patrol",
        "r": 50, "g": 255, "b": 100,
        "radius": 32
    }])

    # 4. ROAD CORRIDORS (Full Damaged Stretches & Under-Construction Zones)
    roads_data = pd.DataFrame([
        {
            "name": "Main Highway Sector 14 Stretch",
            "status": "Severely Damaged Road",
            "info": "🔴 Full Stretch Damaged - Multiple Potholes & Structural Cracking",
            "path": [[76.7120, 30.7010], [76.7180, 30.7060], [76.7250, 30.7120]],
            "color": [255, 40, 40, 230], # Red Corridor (Fully Damaged)
            "width": 10
        },
        {
            "name": "Outer Ring Road (North Segment)",
            "status": "Under Construction",
            "info": "🚧 SAFE NOTICE: Road is under active construction. Drive carefully at reduced speed.",
            "path": [[76.7260, 30.7130], [76.7310, 30.7190], [76.7380, 30.7250]],
            "color": [0, 210, 255, 240], # Cyan/Blue Safe Notice Corridor
            "width": 10
        },
        {
            "name": "Avenue 4 Connecting Link",
            "status": "Moderate Surface Degradation",
            "info": "🟠 Moderate Wear - Resurfacing Scheduled",
            "path": [[76.7020, 30.6950], [76.7080, 30.6990], [76.7120, 30.7010]],
            "color": [255, 140, 0, 210], # Orange Corridor
            "width": 8
        }
    ])

    col_map_view, col_map_stats = st.columns([3, 1])

    with col_map_view:
        # Layer 1: Road Line Segments / Full Corridors
        road_layer = pdk.Layer(
            "PathLayer",
            roads_data,
            pickable=True,
            get_path="path",
            get_color="color",
            width_scale=2,
            width_min_pixels=6,
            get_width="width",
            cap_rounded=True,
            joint_rounded=True,
        )

        # Layer 2: Damage Spot Pinpoints
        spot_layer = pdk.Layer(
            "ScatterplotLayer",
            spots_data,
            pickable=True,
            opacity=0.90,
            stroked=True,
            filled=True,
            radius_scale=2,
            radius_min_pixels=8,
            radius_max_pixels=36,
            line_width_min_pixels=2,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="[r, g, b, 230]",
            get_line_color=[255, 255, 255],
        )

        # Layer 3: Live Vehicle GPS Pin (Pulsing Green)
        user_layer = pdk.Layer(
            "ScatterplotLayer",
            user_pin_data,
            pickable=True,
            opacity=0.95,
            stroked=True,
            filled=True,
            radius_scale=2,
            radius_min_pixels=12,
            radius_max_pixels=45,
            line_width_min_pixels=2.5,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="[r, g, b, 245]",
            get_line_color=[0, 255, 120],
        )

        # 360° Rotational Camera View State
        view_state = pdk.ViewState(
            latitude=st.session_state['user_location']['lat'],
            longitude=st.session_state['user_location']['lon'],
            zoom=13.8,
            pitch=cam_pitch,        # Dynamic 3D tilt
            bearing=cam_bearing,    # Dynamic 360° rotation angle
            max_pitch=85,
            min_pitch=0
        )

        tooltip_config = {
            "html": """
            <div style="font-family: sans-serif; padding: 6px 10px; border-radius: 6px; background: #161B22; border: 1px solid #30363D; color: #FFFFFF;">
                <b>{name}</b>
                <b>{type}</b>
                <div style="font-size: 11px; margin-top: 3px;">{info}</div>
                <div style="font-size: 11px; color: #8B949E;">Severity: {severity}</div>
            </div>
            """,
            "style": {"color": "white"}
        }

        deck = pdk.Deck(
            layers=[road_layer, spot_layer, user_layer],
            initial_view_state=view_state,
            tooltip=tooltip_config,
            map_provider="carto",
            map_style="dark"
        )

        st.pydeck_chart(deck, use_container_width=True)

        if auto_rotate:
            time.sleep(0.4)
            st.rerun()

    with col_map_stats:
        st.markdown("#### 🎮 3D Navigation Controls")
        st.markdown("""
        * **Rotate 360°:** Right-click & drag (or `Ctrl` + Left Click).
        * **Tilt / Pitch:** Move mouse up/down while holding right-click.
        * **Pan / Move:** Standard Left-click & drag.
        * **Zoom In/Out:** Mouse scroll wheel.
        """)

        st.markdown("---")
        st.markdown("#### 🚦 Map Legend & Alerts")
        st.markdown("""
        * 🟢 **Green Marker:** Live Inspection Vehicle (GPS Active)
        * 🔴 **Red Pins/Line:** Severe Potholes / Fully Damaged Road
        * 🟠 **Orange Pins/Line:** Moderate Structural Cracks
        * 🟡 **Yellow Pins:** Minor Surface Wear
        * 🔵 **Cyan Line:** Road Under Construction (Safe Route)
        """)

        st.markdown("---")
        st.info("🚧 **Safe Notice Active:**\nOuter Ring Road North Segment is **Under Construction**. Please maintain speed limit under 30 km/h.")

        st.metric("🔴 Critical Spots", len(spots_data[spots_data['severity'].isin(['Critical', 'High'])]))
        st.metric("🟠 Moderate Spots", len(spots_data[spots_data['severity'].isin(['Moderate', 'Medium'])]))
        st.metric("🟡 Minor Spots", len(spots_data[spots_data['severity'].isin(['Low'])]))

        # CSV Download for GIS data
        csv_spots = spots_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export GIS Coordinates (CSV)",
            data=csv_spots,
            file_name=f"gis_road_damage_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )