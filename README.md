# Real-Time Industrial Anomaly Detection Platform

A comprehensive, production-grade Machine Learning Operations (MLOps) platform designed to ingest, analyze, and flag anomalies across high-frequency industrial data streams in real-time.

![System Architecture](docs/final_report/figures/high_level_arc.png)

## 📖 Comprehensive Final Report
For an incredibly detailed academic breakdown of the algorithms, feature engineering math, system architecture, and empirical test results, please refer to the final project report:
👉 **[FINAL_PROJECT_REPORT.pdf](docs/final_report/FINAL_PROJECT_REPORT.pdf)**

---

## 🚀 Features

*   **Real-Time Inference Engine:** A low-latency FastAPI backend that utilizes a stateful `collections.deque` buffer to extract complex statistical and frequency-domain features (FFT, EWMA) on-the-fly.
*   **Production-Grade MLOps:** Features an active Model Registry (`model_registry.json`) allowing for the one-click hot-swapping of unsupervised algorithms (Isolation Forest, One-Class SVM, LSTM Autoencoders) with zero downtime.
*   **Data Drift Detection:** Continuously calculates the Population Stability Index (PSI) to detect statistical baseline shifts in factory machinery and recommend retraining.
*   **Human-In-The-Loop Alert Center:** A highly responsive React/Vite frontend that enforces a strict lifecycle state machine for generated alerts, requiring qualitative Operator feedback to build a supervised ground-truth dataset.
*   **Chaos Engineering (Synthetic Faults):** Includes a dynamic Stream Simulator capable of injecting Spikes, Gradual Drift, and Sensor Freezes into the live stream to rigorously test detection latency.

## 🛠️ Technology Stack

*   **Backend:** Python 3.10, FastAPI, Uvicorn, asyncio, NumPy, scikit-learn, River (Online ML).
*   **Frontend:** React 18, TypeScript, Vite, TailwindCSS, shadcn/ui, Recharts.
*   **Infrastructure:** Docker, Docker Compose, SQLite (Extensible to PostgreSQL/TimescaleDB), WebSockets.

---

## ⚙️ Installation and Setup

### Method 1: Docker (Recommended)
The easiest way to run the entire multi-modal platform is via Docker Compose.

```bash
# Clone the repository
git clone https://github.com/ahmedmoatasem01/Real-Time-Anomaly-Detection-for-IoT-Sensor-Streams.git
cd Real-Time-Anomaly-Detection-for-IoT-Sensor-Streams

# Build and start the containers
docker compose up --build
```
*   The React Frontend will be available at: `http://localhost:5174`
*   The FastAPI Backend swagger docs will be at: `http://localhost:8000/docs`

### Method 2: Local Python & Node Execution
If you wish to develop locally:

1.  **Start the Backend:**
    ```bash
    python -m venv .venv
    source .venv/Scripts/activate  # (Windows)
    pip install -r requirements.txt
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```

2.  **Start the Frontend:**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

3.  **Start the Live Stream Simulator:**
    ```bash
    # Run this in a 3rd terminal to begin pumping data into the system
    python -m src.streaming.stream_simulator --speed 50 --loop
    ```

---

## 🗺️ Multi-Modal Roadmap

This platform is actively expanding beyond simple 1D telemetry into a unified Asset Center.
*   **Phase 1 (MVP - Complete):** Live Telemetry and Temperature monitoring (Numenta Anomaly Benchmark).
*   **Phase 2 (In Development):** Vibration Health Module. Ingesting 20kHz acoustic/vibration data (NASA Bearing Dataset) using 1D CNN Autoencoders and real-time FFT spectrum visualization.
*   **Phase 3 (Planned):** Visual Inspection Module. Utilizing ResNet50 deep embeddings on the MVTec AD dataset to generate localized defect heatmaps for automated optical inspection.
