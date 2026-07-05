## 21. Frontend Platform Architecture

While Python-based rapid prototyping tools (like Streamlit or Dash) are excellent for data science exploration, they are fundamentally inadequate for a production-grade industrial platform. They suffer from severe latency during high-frequency data ingestion, state management limitations, and rigid UI constraints. 

To deliver a professional, enterprise-grade experience, the user interface was built as a dedicated Single Page Application (SPA) utilizing the React ecosystem.

### 21.1 Core Frontend Technologies
*   **Framework:** React 18 with TypeScript, built utilizing Vite for lightning-fast Hot Module Replacement (HMR). Strict TypeScript typings enforce interface contracts with the backend API.
*   **State Management:** Standard React Hooks (`useState`, `useEffect`) manage localized component state. The WebSocket connection is maintained at the root `App` context, broadcasting data downwards to child components.
*   **Styling (TailwindCSS):** The platform rejects traditional monolithic CSS stylesheets in favor of TailwindCSS. This utility-first framework enables the rapid construction of a dark-mode, premium aesthetic featuring glassmorphism elements, drop shadows, and responsive grid layouts perfectly suited for low-light factory control rooms.
*   **UI Components (`shadcn/ui`):** Instead of utilizing rigid libraries like Material-UI, the project implements `shadcn/ui` (built atop Radix UI primitives). This provides fully accessible, unstyled core components (Dropdowns, Modals, Data Tables) that were heavily customized to match the platform's bespoke design language.
*   **Data Visualization (`Recharts`):** Rendering 65 points of telemetry moving at 50 updates per second will crash a standard DOM. `Recharts` relies on highly optimized SVG rendering. It effortlessly handles the continuous rendering of the sliding window, dynamically painting the anomaly score curve in red when it breaches the threshold line.

### 21.2 The Unified Asset Center Paradigm
The React frontend is architected around the concept of a **Unified Asset Center**. An industrial asset (e.g., "CNC Machine 01") is a complex entity containing multiple modalities:
1.  Spindle Temperature (Telemetry Module)
2.  Spindle Vibration (Vibration Module)
3.  Finished Part Camera (Vision Module)

The frontend routes are designed to wrap these three distinct data pipelines under a single Asset ID context. If the temperature Isolation Forest spikes, OR the vibration Autoencoder detects high reconstruction error, OR the vision MIL loss spikes, a unified Alert is raised against the entire Asset, providing operators with a holistic, single-pane-of-glass view of machine health.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 22. Database Schema and Persistence Layer

A lightweight SQLite database (`anomaly.db`) persists the operational state. While SQLite is utilized for this prototype deployment due to its lack of external daemon dependencies, the SQLAlchemy ORM layer ensures that migrating to PostgreSQL or TimescaleDB for a massive enterprise deployment requires changing only a single connection string.

### 22.1 Entity Relationship Diagram

*(Note: The following ERD illustrates the normalized relationships between raw telemetry, algorithmic models, and human-driven alerts.)*

![Database ERD](figures/db_erd.png)

### 22.2 Schema Definitions
*   **`READINGS` Table:** A high-throughput ledger storing every raw float ingested by the `/predict` endpoint. (Note: In a true production environment, this would be heavily partitioned or migrated to InfluxDB).
*   **`MODELS` Table:** The relational backing for the JSON Model Registry, storing the `uuid`, the boolean `is_production` flag, and the absolute filepath to the serialized artifact.
*   **`ALERTS` Table:** The most complex entity. It links the mathematical breach to the human workflow. It forces strict string enumerations for both `severity` and `status`, preventing operator error.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 23. Backend REST API Documentation

The FastAPI backend exposes a strictly typed REST architectural layer. Every endpoint utilizes Pydantic models for rigorous request/response validation.

### 23.1 Real-Time Ingestion endpoint
*   **Route:** `POST /predict`
*   **Purpose:** The primary ingestion vector for IoT edge devices.
*   **Request Payload Example:**
    ```json
    {
      "sensor_id": "T-01",
      "timestamp": "2026-07-05T14:32:01.000Z",
      "value": 85.4
    }
    ```
*   **Response Payload Example (200 OK):**
    ```json
    {
      "status": "success",
      "score": 0.84,
      "is_anomaly": true,
      "alert_id": 402,
      "message": "Critical threshold breached."
    }
    ```

### 23.2 WebSocket Streaming Interface
*   **Route:** `WS /ws/stream`
*   **Purpose:** Persistent bidirectional connection for pushing live UI updates.
*   **Broadcast Payload Example:**
    ```json
    {
      "type": "telemetry_update",
      "sensor_id": "T-01",
      "value": 85.4,
      "score": 0.84,
      "features": {
        "rolling_mean_65": 50.1,
        "rolling_std_65": 14.2
      },
      "active_alerts": [
        {
          "id": 402,
          "severity": "critical"
        }
      ]
    }
    ```

### 23.3 Alert Lifecycle Management
*   **Route:** `POST /alerts/{alert_id}/resolve`
*   **Purpose:** Submits human operator feedback and transitions the alert to a terminal state.
*   **Request Payload Example:**
    ```json
    {
      "resolution_status": "resolved",
      "operator_note": "Technician replaced the frayed V-belt. Machine is operating at nominal parameters."
    }
    ```
*   **Response Payload Example:**
    ```json
    {
      "id": 402,
      "status": "resolved",
      "operator_note": "Technician replaced...",
      "resolved_at": "2026-07-05T14:45:00.000Z"
    }
    ```

### 23.4 Incident Report PDF Generator
*   **Route:** `GET /reports/incident/{alert_id}`
*   **Purpose:** Generates a professional, legally-compliant PDF report detailing the anomaly event.
*   **Internal Logic:** The backend queries the SQLite DB for the alert. It then queries the `READINGS` table to extract the 10 data points immediately preceding the alert timestamp. It utilizes the `ReportLab` Python library to dynamically draw tables, bold fonts, and inject the operator's notes onto a PDF canvas.
*   **Response:** `application/pdf` binary stream.

### 23.5 Synthetic Fault Injection
*   **Route:** `POST /faults/inject`
*   **Purpose:** Allows the frontend Demo Control Panel to dynamically corrupt the data stream.
*   **Request Payload Example:**
    ```json
    {
      "fault_type": "drift",
      "duration_seconds": 60,
      "magnitude": 0.5
    }
    ```

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 24. Future Roadmap: The Vibration Health Module

While temperature (Telemetry) is excellent for detecting friction, mechanical systems emit high-frequency acoustic vibrations long before thermal changes manifest. The integration of the NASA Bearing dataset represents the next massive evolutionary step for the platform.

### 24.1 Edge Chunking vs Cloud Streaming
Because the NASA dataset operates at 20kHz, streaming 20,000 floats per second over an HTTP REST API will instantly crash the backend due to TCP overhead. 
The architectural design dictates that an IoT Edge device (e.g., a Raspberry Pi attached to the machine) must buffer 1-second chunks of data. It performs the Fast Fourier Transform (FFT) locally on the edge, and only sends the resulting 1xN feature vector to the cloud API.

### 24.2 The Vibration UI
The React frontend will feature a dedicated Vibration Lab. Instead of a standard line chart, it will render the Frequency Spectrum. The X-axis represents Frequency (Hz), and the Y-axis represents Amplitude. Operators will visually identify harmonic spikes at the specific Ball Pass Frequency Outer-Race (BPFO) or Ball Pass Frequency Inner-Race (BPFI) mathematical fault frequencies.

## 25. Future Roadmap: Visual Inspection Module

Automated Optical Inspection (AOI) requires a radical departure from sliding-window time-series logic.

### 25.1 ResNet50 Deep Embeddings
Utilizing the MVTec AD dataset, high-resolution camera frames of products on an assembly line are captured. A pre-trained Convolutional Neural Network (ResNet50) is used as a feature extractor. The massive 3D image tensor is compressed into a dense 1D vector embedding. This embedding is then passed to an Isolation Forest, which detects if the visual structure deviates from the normal products.

### 25.2 CNN Autoencoder Heatmaps
For explainability, a secondary Convolutional Autoencoder is trained. By calculating the pixel-wise Mean Squared Error (MSE) between the raw camera frame and the Autoencoder's reconstruction, the UI will dynamically generate a red overlay "Heatmap". This explicitly points the human operator to the exact physical location of the scratch, dent, or contamination on the product.
