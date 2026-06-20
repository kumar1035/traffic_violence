

import os
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import database
from detector import TrafficDetector, draw_detections
from preprocessor import preprocess_pipeline
from violations import run_all_checks
from ocr import PlateReader, blur_faces, face_bbox_from_pose
from challan import generate_challan_pdf
import analytics


# ---------- Page config ----------
st.set_page_config(
    page_title="TrafficVision AI",
    page_icon="🚦",
    layout="wide",
)

database.init_db()

EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

ZONE_OPTIONS = ["MG Road", "Silk Board", "Hebbal Flyover", "Whitefield",
                "Electronic City", "Marathahalli", "Other / Unspecified"]


# ---------- Cached model loading ----------
@st.cache_resource
def load_detector():
    return TrafficDetector()


@st.cache_resource
def load_plate_reader():
    return PlateReader(gpu=True)



def process_image(image_bgr, mode, conf_threshold, zone, source="camera"):
    """
    Runs the full pipeline on one image: preprocess -> detect -> violations
    -> OCR -> face blur -> save evidence -> log to DB -> generate challan.
    Returns (annotated_image, violations_list, processing_time)
    """
    start_time = time.time()

    detector = load_detector()
    plate_reader = load_plate_reader()

    processed = preprocess_pipeline(image_bgr, mode=mode)

    results = detector.detect_all(processed, conf_threshold=conf_threshold)
    objects, poses, helmets = results["objects"], results["poses"], results["helmets"]

    print(f"\n[DEBUG] conf_threshold={conf_threshold}")
    print(f"[DEBUG] objects found: {len(objects)}")
    print(f"[DEBUG] poses found: {len(poses)}")
    print(f"[DEBUG] helmets found: {len(helmets)}")
    for h in helmets:
        print(f"[DEBUG]   helmet: {h['class_name']} conf={h['confidence']} bbox={h['bbox']}")
    motos_debug = [o for o in objects if o["class_name"] == "motorcycle"]
    print(f"[DEBUG] motorcycles found: {len(motos_debug)}")
    for m in motos_debug:
        print(f"[DEBUG]   motorcycle bbox={m['bbox']}")

    violations = run_all_checks(
        objects, poses, helmets,
        stop_line_y=None, signal_is_red=False,
        conf_filter=conf_threshold,
    )

    print(f"[DEBUG] violations found: {len(violations)}\n")

  
    annotated = draw_detections(processed, objects)
    helmet_colors = {"With Helmet": (0, 255, 0), "Without Helmet": (0, 0, 255)}
    for h in helmets:
        color = helmet_colors.get(h["class_name"], (255, 255, 0))
        annotated = draw_detections(annotated, [h], color=color)

    for v in violations:
        x1, y1, x2, y2 = v["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(annotated, v["type"], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
    face_boxes = []
    for pose in poses:
        fb = face_bbox_from_pose(pose["keypoints"])
        if fb:
            face_boxes.append(fb)
    annotated_blurred = blur_faces(annotated, face_boxes) if face_boxes else annotated

    
    vehicles = [o for o in objects if o["class_name"] in ("car", "motorcycle", "bus", "truck")]
    enriched_violations = []

    for v in violations:
        plate_result = {"text": None, "confidence": 0.0}
        
        if vehicles:
            nearest = min(vehicles, key=lambda veh: _bbox_distance(v["bbox"], veh["bbox"]))
            plate_result = plate_reader.extract_plate_from_vehicle(processed, nearest["bbox"])

        plate_number = plate_result.get("text")
        is_repeat, repeat_count, _ = database.check_repeat_offender(plate_number) if plate_number else (False, 0, [])

        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        evidence_path = os.path.join(EVIDENCE_DIR, f"{v['type']}_{timestamp_str}.jpg")
        cv2.imwrite(evidence_path, annotated_blurred)

        
        challan_id, pdf_path = generate_challan_pdf(
            violation_type=v["type"],
            plate_number=plate_number,
            confidence=v["confidence"],
            severity=v["severity"],
            zone=zone,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason=v["reason"],
            evidence_image_path=evidence_path,
            is_repeat_offender=is_repeat,
            repeat_count=repeat_count,
        )

        # Log to database
        database.insert_violation(
            violation_type=v["type"],
            confidence=v["confidence"],
            severity=v["severity"],
            zone=zone,
            plate_number=plate_number,
            plate_confidence=plate_result.get("confidence", 0.0),
            reason=v["reason"],
            evidence_image_path=evidence_path,
            challan_id=challan_id,
            challan_pdf_path=pdf_path,
            source=source,
        )

        enriched_violations.append({
            **v,
            "plate_number": plate_number,
            "is_repeat_offender": is_repeat,
            "repeat_count": repeat_count,
            "challan_id": challan_id,
            "challan_pdf_path": pdf_path,
        })

    processing_time = time.time() - start_time
    return annotated_blurred, enriched_violations, processing_time


def _bbox_distance(bbox_a, bbox_b):
    ax = (bbox_a[0] + bbox_a[2]) / 2
    ay = (bbox_a[1] + bbox_a[3]) / 2
    bx = (bbox_b[0] + bbox_b[2]) / 2
    by = (bbox_b[1] + bbox_b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# ---------- Sidebar ----------
st.sidebar.title("🚦 TrafficVision AI")
st.sidebar.caption("AI-driven traffic violation detection & enforcement intelligence")

page = st.sidebar.radio(
    "Navigate",
    ["Live Demo", "Citizen Report", "Evidence Log", "Analytics", "Performance Metrics"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
preprocessing_mode = st.sidebar.selectbox("Preprocessing mode", ["normal", "night", "rain"])
selected_zone = st.sidebar.selectbox("Zone / Location", ZONE_OPTIONS)


# ---------- Page: Live Demo ----------
if page == "Live Demo":
    st.title("Live Violation Detection")
    st.caption("Upload a traffic image to detect violations in real time.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("Running detection pipeline..."):
            annotated, violations, proc_time = process_image(
                image_bgr, preprocessing_mode, conf_threshold, selected_zone
            )

        with col2:
            st.subheader("Annotated Result")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.caption(f"Processed in {proc_time:.2f}s")

        if violations:
            st.success(f"{len(violations)} violation(s) detected")
            for v in violations:
                with st.expander(f"⚠️ {v['type'].replace('_', ' ').title()} — Severity {v['severity']}/10"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Confidence:** {v['confidence']:.1%}")
                        st.markdown(f"**Plate:** {v['plate_number'] or 'Not identified'}")
                        st.markdown(f"**Challan ID:** {v['challan_id']}")
                        if v["is_repeat_offender"]:
                            st.error(f"🚨 Repeat offender — {v['repeat_count']} prior violations")
                    with c2:
                        st.markdown("**Why was this flagged?**")
                        st.info(v["reason"])

                    if os.path.exists(v["challan_pdf_path"]):
                        with open(v["challan_pdf_path"], "rb") as f:
                            st.download_button(
                                "📄 Download E-Challan PDF",
                                f,
                                file_name=f"{v['challan_id']}.pdf",
                                mime="application/pdf",
                                key=f"dl_{v['challan_id']}",
                            )
        else:
            st.info("No violations detected in this image.")


# ---------- Page: Citizen Report ----------
elif page == "Citizen Report":
    st.title("Citizen Violation Reporting")
    st.caption("Spotted a traffic violation? Upload a photo and we'll process it the same way our cameras do.")

    st.warning("📸 By uploading, you confirm this photo was taken in a public space and depicts a genuine traffic violation.")

    uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png", "webp"], key="citizen_upload")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.button("Submit Report"):
            with st.spinner("Processing your report..."):
                annotated, violations, proc_time = process_image(
                    image_bgr, preprocessing_mode, conf_threshold, selected_zone,
                    source="citizen_report",
                )

            if violations:
                st.success(f"Thank you. {len(violations)} violation(s) logged from your report.")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.info("No clear violations were detected in this photo.")


# ---------- Page: Evidence Log ----------
elif page == "Evidence Log":
    st.title("Evidence Log")

    df = analytics.evidence_log_table(limit=200)
    if df.empty:
        st.info("No violations logged yet. Process some images in Live Demo first.")
    else:
        type_filter = st.multiselect("Filter by type", df["Type"].unique().tolist())
        if type_filter:
            df = df[df["Type"].isin(type_filter)]

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df)} record(s)")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download as CSV", csv, "violations_log.csv", "text/csv")

    st.markdown("---")
    st.subheader("🚨 Repeat Offenders")
    offenders_df = analytics.repeat_offenders_table(min_violations=2)
    if offenders_df.empty:
        st.caption("No repeat offenders logged yet.")
    else:
        st.dataframe(offenders_df, use_container_width=True, hide_index=True)


# ---------- Page: Analytics ----------
elif page == "Analytics":
    st.title("Enforcement Analytics")

    stats = database.get_summary_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Violations", stats["total_violations"])
    c2.metric("Unique Plates Flagged", stats["unique_plates_flagged"])
    c3.metric("Avg. Severity", f"{stats['avg_severity']}/10")

    col1, col2 = st.columns(2)
    with col1:
        pie = analytics.violation_type_pie_chart()
        if pie:
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("No data yet.")

    with col2:
        gauge = analytics.severity_gauge(stats["avg_severity"])
        st.plotly_chart(gauge, use_container_width=True)

    bar = analytics.zone_risk_bar_chart()
    if bar:
        st.plotly_chart(bar, use_container_width=True)
        st.caption("Zones ranked by total risk-weighted severity — use this to prioritize patrol deployment.")
    else:
        st.info("No zone data yet.")

    hourly = analytics.hourly_trend_chart()
    if hourly:
        st.plotly_chart(hourly, use_container_width=True)


# ---------- Page: Performance Metrics ----------
elif page == "Performance Metrics":
    st.title("Model Performance Metrics")
    st.caption("Evaluation results from the fine-tuned helmet detection model on a held-out validation set.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", "81.8%")
    c2.metric("Recall", "73.3%")
    c3.metric("mAP50", "78.2%")
    c4.metric("mAP50-95", "37.1%")

    st.markdown("---")
    st.subheader("Per-Class Performance")
    st.table({
        "Class": ["With Helmet", "Without Helmet"],
        "Precision": ["86.6%", "77.0%"],
        "Recall": ["82.6%", "64.0%"],
        "mAP50": ["87.2%", "69.3%"],
    })

    st.caption(
        "Trained on a public Roboflow helmet dataset (126 validation images), "
        "fine-tuned from YOLOv8s. Inference speed: ~5.6ms/image on RTX 3050."
    )


st.sidebar.markdown("---")
st.sidebar.caption("TrafficVision AI — Hackathon Prototype")