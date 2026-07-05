<div style="text-align: center; margin-top: 150px; margin-bottom: 150px;">

# FINAL PROJECT REPORT

<br><br>

# Real-Time Industrial Anomaly Detection Platform
### A Multi-Modal Machine Learning Architecture for Continuous IoT Sensor Monitoring

<br><br><br>

**Author:** AI Agent  
**Institution:** Professional Development Program  
**Date:** July 2026  
**Version:** 1.0.0  
**Repository:** Built on the *Real-Time Anomaly Detection for IoT Sensor Streams* repository.

</div>

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 2. Executive Summary

The advent of the Industrial Internet of Things (IIoT) has ushered in a transformative era of data-rich manufacturing, predictive maintenance, and autonomous operational intelligence. As modern factory floors are increasingly equipped with high-fidelity sensors measuring everything from spindle temperature to acoustic emissions, operators are inundated with telemetry. However, this massive influx of streaming data poses significant analytical challenges. Traditional Supervisory Control and Data Acquisition (SCADA) systems rely almost entirely on static, human-defined thresholds—a paradigm that is mathematically incapable of identifying the subtle, non-linear degradation signatures that precede catastrophic mechanical failure.

The **Real-Time Industrial Anomaly Detection Platform** was developed as a comprehensive, production-grade Machine Learning Operations (MLOps) system designed to solve this exact problem. It is built to seamlessly ingest, process, analyze, and flag anomalies across high-frequency industrial data streams with millisecond latency. 

This project delivers a robust, real-time pipeline that dramatically transcends traditional dashboarding logic by embedding active Machine Learning algorithms directly into the critical data path. Utilizing the rigorously evaluated Numenta Anomaly Benchmark (NAB) Machine Temperature dataset, the system empirically establishes a baseline of normal operating behavior. Instead of reacting to hard limits, the system extracts complex rolling statistical features, frequency-domain transformations, and temporal derivatives in real-time. It then employs a suite of unsupervised machine learning models—primarily an Isolation Forest algorithm that proved superior during empirical testing—to calculate an ongoing, continuous anomaly score for every incoming reading.

When calculated scores breach algorithmically optimized thresholds, the platform transitions from passive detection to active human-in-the-loop management. It initiates a structured Alert Lifecycle, empowering human operators to acknowledge, investigate, and resolve issues within a highly responsive React-based interface. Crucially, the platform captures operator feedback, creating a qualitative loop that validates the quantitative machine learning flags.

Moving beyond simple detection mechanics, the system acts as a complete MLOps ecosystem. It continuously calculates Data Drift metrics utilizing the Population Stability Index (PSI) to track long-term baseline shifts in factory environments. When drift reaches critical levels, the system directs users to a comprehensive Retraining Center, featuring a dynamic Model Registry that supports the hot-swapping of offline-trained candidate models into live production. Ultimately, the resulting architecture provides a scalable, multi-modal framework that allows for the independent yet unified management of Telemetry, Vibration, and Visual Inspection modules within a centralized Asset Center.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 3. Business Problem and Motivation

### 3.1 The Cost of Industrial Downtime
Unplanned industrial downtime represents one of the most significant financial sinks in the modern global economy, costing manufacturers billions of dollars annually. When a critical asset—such as a CNC machine spindle, an injection molding press, or a large-scale turbine—experiences catastrophic failure, the financial repercussions scale exponentially. The costs are not merely localized to the replacement value of the damaged component; they encompass immediate production halts, missed shipping SLAs, compounding supply chain delays, and severe safety risks to the operators on the floor. 

Reactive maintenance—the practice of running machinery until it physically breaks and then executing emergency repairs—is no longer financially viable in highly optimized, just-in-time manufacturing environments. Preventive maintenance—replacing parts on a rigid chronological schedule regardless of their actual health—often results in perfectly functional, expensive components being discarded prematurely. 

### 3.2 The Flaws of Static SCADA Thresholds
To combat this, the industry shifted toward predictive maintenance via sensor monitoring. However, the standard approach in legacy SCADA systems involves operators setting static upper and lower control limits (e.g., "Trigger a critical alarm if the temperature exceeds 100°C"). This approach suffers from fatal mathematical and operational flaws:

1.  **Missed Early Warnings (High False Negatives):** Mechanical degradation is rarely an instantaneous leap to failure. It often manifests as subtle shifts in variance, localized harmonic distortions, or slight changes in the rate-of-change long before a static thermal threshold is breached. By the time the temperature hits 100°C, the inner race of a bearing may have already shattered.
2.  **Alert Fatigue (High False Positives):** Simple thresholds generate massive amounts of false alarms during normal, operational fluctuations (e.g., a machine naturally running hotter during a heavy-load summer shift). When operators are bombarded by hundreds of non-critical alerts per day, they develop "Alert Fatigue," routinely ignoring or muting the very alarms designed to prevent disasters.

### 3.3 The Need for Real-Time Machine Learning
The fundamental motivation behind this project is to replace static rules with dynamic, context-aware Machine Learning. 

By employing unsupervised machine learning over a rolling temporal window, the platform learns the complex, multi-dimensional correlations between moving averages, signal volatility, and spectral components. It detects the *contextual anomalies* that precede failure. For example, a temperature of 85°C might be perfectly normal if the machine has been running for 8 hours, but highly anomalous if it has only been running for 5 minutes. Static thresholds cannot grasp this context; Machine Learning can.

Furthermore, dashboards alone are insufficient. A dashboard that simply charts a spike provides no operational guidance. The motivation of this platform is to provide **Decision Support**. By formalizing an "Alert Lifecycle" and combining it with automated explainability (Feature Attribution), the platform treats anomaly detection as a collaborative human-AI workflow, rather than a black-box alarm system.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 4. Project Objectives

The Real-Time Industrial Anomaly Detection Platform was architected from the ground up with a series of distinct, overlapping objectives spanning data science, software engineering, and product design.

### 4.1 Primary Operational Objective
*   Design, build, and deploy an end-to-end, real-time streaming pipeline that accurately identifies equipment degradation before catastrophic failure, minimizing both False Alarms and Missed Detections.

### 4.2 Machine Learning & Data Science Objectives
*   **Algorithmic Benchmarking:** Implement, rigorously evaluate, and benchmark a diverse suite of unsupervised anomaly detection models, encompassing classical statistical methods (Rolling Z-score, Elliptic Envelope), distance/density-based models (LOF), boundary-based models (One-Class SVM), ensemble tree methods (Isolation Forest), deep learning architectures (LSTM Autoencoders), and online streaming models (River HalfSpaceTrees).
*   **Feature Engineering Execution:** Mathematically formulate and extract complex temporal and frequency-domain features (EWMA, Kurtosis, FFT) from raw, noisy, univariate data streams in real-time.
*   **Threshold Optimization:** Develop a dynamic, F1-maximizing threshold selection methodology that relies strictly on validation distributions to prevent data leakage.

### 4.3 Software Engineering & Architecture Objectives
*   **High-Throughput Streaming:** Construct a low-latency, highly scalable backend leveraging FastAPI, capable of handling rapid HTTP POST ingestion and maintaining stateful, rolling memory buffers for thousands of concurrent sensors.
*   **Asynchronous Communication:** Implement a robust WebSocket broadcasting layer to push real-time anomaly scores and telemetry directly to the client without aggressive client-side polling.
*   **Data Integrity:** Design a normalized SQLite relational database schema (extensible to PostgreSQL) to persist telemetry history, alert states, and complex model metadata.

### 4.4 MLOps & Lifecycle Objectives
*   **Model Registry Implementation:** Create a deterministic Model Registry that tracks serialized artifacts (`.pkl`), optimal thresholds (`.json`), feature signatures, and historical test metrics for every trained model.
*   **Automated Retraining Flow:** Build an operational loop that tracks statistical Data Drift (via PSI), flags baseline degradation, and guides the operator through an offline retraining and candidate promotion workflow.
*   **Continuous Testing:** Implement synthetic fault injection endpoints to simulate extreme edge-cases (e.g., Sensor Stuck, Gradual Drift) and empirically verify system response latency.

### 4.5 Product & Frontend Objectives
*   **Premium User Experience:** Develop a highly responsive, aesthetically premium React/TypeScript interface utilizing TailwindCSS and `shadcn/ui`, moving beyond basic Streamlit prototypes.
*   **Workflow Integration:** Build an Alert Center that enforces a strict state-machine lifecycle (`new` $\rightarrow$ `acknowledged` $\rightarrow$ `resolved`), forcing human accountability and capturing qualitative ground-truth feedback.
*   **Reporting Excellence:** Enable the programmatic generation of professional PDF Incident Reports utilizing ReportLab, bridging the gap between algorithmic detection and corporate compliance.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 5. Scope of the Project

Given the massive complexity of Industrial AI, this platform is designed to be highly extensible. The scope is explicitly defined into active capabilities, roadmap features, and intentional exclusions to maintain focus.

### 5.1 Currently Implemented Scope (The Active MVP)
The core repository currently contains a fully functional, end-to-end MVP tailored specifically for low-frequency telemetry.
*   **Live Telemetry Pipeline:** Complete ingestion, preprocessing, real-time feature engineering, and inference using the NAB Machine Temperature dataset.
*   **Backend Architecture:** The FastAPI application, WebSocket broadcasting, and SQLite alert/telemetry persistence layers.
*   **Model Registry & MLOps:** The JSON-backed model artifact tracker, offline evaluation reports, and the Population Stability Index (PSI) drift tracking algorithms.
*   **Frontend Ecosystem:** A comprehensive React SPA including the Live Monitor, System Health dashboard, Model Lab, Data Explorer, and the interactive Alert Center.
*   **Synthetic Fault Injection:** A demo-critical system capable of dynamically altering the simulated data stream to inject Spikes, Drift, Stuck sensors, and Noise Bursts to test UI and algorithmic response.
*   **PDF Generation:** The automated compilation of Incident Reports upon alert resolution.

### 5.2 Advanced Roadmap Scope (Multi-Modal Extensions)
Industrial environments require more than just temperature monitoring. The architecture has been explicitly designed to support two massive, multi-modal extensions currently slated as roadmap items:
*   **Vibration Module (High-Frequency Analytics):** Processing 20kHz accelerometer data utilizing the NASA Bearing dataset. This requires an entirely different edge-chunking architecture, relying on FFT spectrum analysis, Envelope Demodulation, and 1D CNN Autoencoders to predict Remaining Useful Life (RUL).
*   **Visual Inspection Module (Spatial Analytics):** Utilizing the MVTec AD image dataset to detect manufacturing defects (e.g., scratches on a pill, bent wire) on the assembly line. This relies on heavy ResNet50 embeddings, Isolation Forests for normal-only clustering, and CNN Autoencoder spatial heatmaps for explainability.

### 5.3 Intentional Exclusions
To maintain portability and ease of deployment for demonstration and grading purposes, several enterprise-level systems were intentionally excluded:
*   **Distributed Message Brokers:** In a true factory, sensors would push to an MQTT broker, which would feed into Apache Kafka. For this project, a Python-based Stream Simulator and HTTP/WebSocket endpoints simulate this behavior, removing the need for heavy Java/Erlang dependencies.
*   **Cloud-Native Microservices:** Deployments to AWS EKS or Azure IoT Hub are excluded. The entire system is portable via a localized Docker Compose stack, ensuring it runs reliably on standard hardware.
*   **Heavy Identity Providers (IdP):** Full OAuth2/OIDC integration (e.g., Okta, Auth0) is excluded, though the routing is designed to easily accept role-based access control (RBAC) middleware in the future.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 6. System Overview

The Real-Time Industrial Anomaly Detection Platform is conceptually divided into four horizontal layers: Data Ingestion/Simulation, the FastAPI Backend Engine, the Machine Learning Storage/Registry, and the React Client Application.

### 6.1 The Operational Flow
The primary operational flow begins at the edge. Because real industrial machinery is not continuously available, a Python-based **Stream Simulator** reads historical test data row-by-row and fires HTTP POST requests to the FastAPI backend at a configurable speed multiplier. Before leaving the simulator, the data passes through the **Synthetic Fault Injector**, which can dynamically corrupt the data (e.g., multiplying the value by 3x to simulate a spike) based on operator commands.

Once the payload arrives at the `/predict` endpoint, it enters a highly optimized, stateful **Sliding Buffer**. The backend maintains a unique buffer queue (maximum length of 65 readings) for every distinct `sensor_id`. As soon as the buffer is full, the system executes real-time **Feature Engineering**. It extracts localized context—such as the rolling standard deviation of the last 15 points, the EWMA, and the FFT dominant frequency—transforming a single noisy temperature float into a stable 1xN feature vector.

This feature vector is immediately passed to the **Production ML Model** (currently the Isolation Forest). The model calculates a continuous anomaly score. The backend compares this score against the pre-calculated, validation-optimized **Threshold**. If the score exceeds the threshold, the system calculates the severity magnitude and automatically persists an Alert record in the SQLite **Database**. 

Finally, regardless of whether an alert was triggered, the raw reading, the engineered features, the anomaly score, and any active alerts are serialized and broadcast via **WebSockets** to all connected React clients. The frontend parses this stream, seamlessly updating the SVG Recharts graphs, the Alert Kanban boards, and the Drift PSI gauges without requiring a hard page refresh.

### 6.2 High-Level System Architecture Figure

*(Note: The following diagram illustrates the overarching data flow of the MVP Telemetry pipeline.)*

![High-Level System Architecture](figures/high_level_arc.png)

As demonstrated in the architecture figure above, the system relies on a clean separation of concerns. The Machine Learning models (Isolation Forest, River ADWIN) act as independent mathematical functions completely decoupled from the WebSocket and REST routing layers. This decoupling enables the hot-swapping capability of the Model Registry, allowing an operator to seamlessly replace the `Isolation Forest` with a newly trained `One-Class SVM` in real-time, without dropping a single HTTP packet or interrupting the websocket stream.

The persistence layer relies on SQLite for relational metadata (Alerts, Model Metadata, Retraining Runs) and flat-file JSON/PKL artifacts for mathematical matrices and thresholds. The React frontend is entirely stateless, drawing its single source of truth entirely from the FastAPI service layer.
