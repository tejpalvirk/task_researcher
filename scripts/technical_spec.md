# Technical Specification: Real-time Anomaly Detection System

## 1. Architecture

*   **Style:** Microservices Architecture.
*   **Core Services:**
    *   `data-ingestor`: Consumes upstream data, validates, publishes internally.
    *   `feature-engineer`: Computes derived features from transactions.
    *   `stat-detector`: Applies statistical anomaly detection models.
    *   `ml-detector`: Applies ML anomaly detection models.
    *   `rule-engine`: Applies configurable business rules.
    *   `anomaly-aggregator`: Combines detection outputs, stores results.
    *   `api-server`: Provides REST API for dashboard and external queries.
    *   `dashboard-frontend`: Web user interface.
    *   `alerter`: Sends notifications based on detected anomalies.
*   **Communication:** Apache Kafka for asynchronous messaging between services. REST (or potentially gRPC) for synchronous requests (e.g., API server to database).
*   **Deployment:** Docker containers orchestrated by Kubernetes (AWS EKS / Google GKE / Azure AKS).

## 2. Technology Stack

*   **Primary Language:** Python 3.11+
*   **Messaging:** Apache Kafka (Managed service preferred, e.g., Confluent Cloud, MSK).
*   **Database:** PostgreSQL with TimescaleDB extension for time-series anomaly data. Redis for caching/rate-limiting/short-term state.
*   **API Framework:** FastAPI.
*   **Data Processing/ML:** Pandas, NumPy, Scikit-learn, PyTorch/TensorFlow (for Autoencoders), MLflow (for model lifecycle). Pydantic for data validation/schemas.
*   **Frontend:** React or Vue.js (TBD). Data visualization library (e.g., D3.js, Chart.js).
*   **Infrastructure Provisioning:** Terraform or Pulumi.
*   **CI/CD:** GitHub Actions or GitLab CI.
*   **Monitoring:** Prometheus for metrics collection, Grafana for visualization, Loki for logging aggregation (optional: Jaeger/Tempo for tracing).
*   **Containerization:** Docker.

## 3. Data Schemas & Storage

*   Transaction and Anomaly schemas defined using Pydantic models.
*   TimescaleDB hypertables used for efficient storage and querying of transaction features and detected anomalies.
*   Consider partitioning strategies in TimescaleDB based on time and potentially user ID/merchant ID.
*   ML models stored and versioned using MLflow Model Registry.

## 4. Anomaly Detection Logic

*   **Statistical:** Implement basic Z-score and Interquartile Range (IQR) detectors initially.
*   **ML:** Prioritize Isolation Forest for efficiency. Explore Autoencoders for complex pattern detection. Use MLflow for training pipeline orchestration and model serving integration.
*   **Rules:** Implement a simple rule engine capable of evaluating conditions on feature dictionaries. Rules defined in a configuration file (e.g., YAML) or database.

## 5. API Design

*   RESTful API provided by `api-server`.
*   Endpoints for:
    *   Querying anomalies (GET `/anomalies` with filtering/pagination).
    *   Getting anomaly details (GET `/anomalies/{id}`).
    *   System status (GET `/status`).
    *   (Admin) Managing rules/configurations.
*   Authentication via JWT Bearer tokens.

## 6. Testing Strategy

*   **Unit Tests:** Pytest for individual functions and classes within services. Target >80% coverage.
*   **Integration Tests:** Test interactions between services via Kafka topics and APIs. Use test containers for dependencies (Kafka, DB).
*   **End-to-End Tests:** Simulate data flow from ingestion to alerting (potentially using a subset of services).
*   **Performance Tests:** Locust or k6 to simulate load on the API and data processing pipeline.
*   **Contract Tests:** (Optional) Pact or similar to ensure service interfaces remain compatible.

## 7. Security Considerations

*   Secure communication between services (e.g., TLS for Kafka, HTTPS for APIs).
*   Input validation at service boundaries.
*   Authentication and authorization for API and Dashboard access.
*   Secrets management (e.g., HashiCorp Vault, cloud provider secrets manager).
*   Regular dependency scanning.