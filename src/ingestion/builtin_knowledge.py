"""
Built-in Knowledge Base
Structured excerpts from published research papers by Md Arifur Rahman.
Used as default knowledge when no PDF documents are provided.

Papers covered:
1. CNN-LSTM-SW for Railroad Anomaly Detection (Elsevier GEITS 2024)
2. DAS-based Railroad CM with GRU/LSTM (SPIE JARS 2024)
3. Review of DAS Applications for Railroad CM (MSSP, Elsevier)
4. HMM-RL for WAAM Manufacturing Optimization (Springer 2026)
"""

from src.retrieval.rag_pipeline import DocumentChunk


def get_builtin_chunks() -> list[DocumentChunk]:
    """Return structured knowledge chunks from published papers."""

    papers = [

        # ── Paper 1: CNN-LSTM-SW ──────────────────────────────────────────────
        {
            "title": "Remote condition monitoring of rail tracks using DAS: A deep CNN-LSTM-SW model",
            "authors": "Rahman MA, Jamal S, Taheri H",
            "year": "2024",
            "doi": "10.1016/j.geits.2024.100178",
            "venue": "Green Energy and Intelligent Transportation, Elsevier",
            "domain": "railroad_anomaly_detection",
            "chunks": [
                "The CNN-LSTM-SW model combines convolutional neural networks (CNN) for spatial "
                "feature extraction, long short-term memory (LSTM) networks for temporal pattern "
                "learning, and a sliding window (SW) post-processing step that corrects "
                "point-to-point misclassifications along the railroad track. The model was "
                "validated on HTL loop DAS data from the AAR/TTCI test facility in Pueblo, CO, "
                "covering 4.16 km of fiber-optic instrumented track.",

                "The sliding window (SW) correction method addresses a key limitation of "
                "CNN-LSTM: boundary misclassifications between adjacent condition classes "
                "(e.g., TP/AC2 confusion). A majority-vote window of configurable size is "
                "applied to the CNN-LSTM output predictions. For each position i, the window "
                "considers neighboring predictions and assigns the majority class if it exceeds "
                "a threshold fraction. This dramatically reduces isolated misclassifications "
                "while preserving true anomaly boundaries.",

                "Four condition classes are defined for the HTL loop dataset: NC (Normal "
                "Condition) — background rail and environmental noise; TP (Train Position) — "
                "acoustic signal from a passing train; AC1 (Anomaly Class 1) — light defect "
                "or wheel flat; AC2 (Anomaly Class 2) — rail joint or heavier anomaly. The "
                "class distribution is highly imbalanced: NC dominates at ~93% of all samples, "
                "making SMOTE oversampling essential for model training.",

                "Three feature domains are extracted from DAS multi-channel signals: time "
                "domain (mean, Pearson skewness, kurtosis), frequency domain (band energy "
                "ratio, spectral kurtosis, spectral contrast via FFT), and time-frequency "
                "domain (STFT-based spectral mean, std, max, median). The combined T&FD "
                "feature set achieved the best model performance with 97% train position "
                "detection rate, outperforming TD-only (89%) and FD-only (88%) configurations.",

                "Model performance comparison: CNN-LSTM-SW (proposed) achieved 97% train "
                "position detection with corrected misclassifications — best overall. CNN-LSTM "
                "baseline achieved 97% detection but with uncorrected boundary errors. GRU "
                "baseline achieved 94% detection. LSTM baseline achieved 93% detection. "
                "The ROC AUC scores were: AC1=1.00, NC=0.99, TP=0.97, AC2=0.95.",
            ]
        },

        # ── Paper 2: GRU/LSTM DAS ─────────────────────────────────────────────
        {
            "title": "Railroad condition monitoring with distributed acoustic sensing: "
                     "deep learning methods for condition detection",
            "authors": "Rahman MA, Kim J, Dababneh F, Taheri H",
            "year": "2024",
            "doi": "10.1117/1.JRS.18.016512",
            "venue": "Journal of Applied Remote Sensing, SPIE",
            "domain": "railroad_condition_monitoring",
            "chunks": [
                "Distributed acoustic sensing (DAS) using fiber optic cables provides a "
                "cost-effective and scalable technique for condition monitoring (CM) of "
                "railroad infrastructure over extensive distances. The Rayleigh backscattering "
                "principle converts acoustic vibrations from passing trains into measurable "
                "optical signals. The iDAS system (Silixa) used in this study has a gauge "
                "length of 10m, sensing range of 45km, and sampling frequency up to 100kHz.",

                "The GRU (Gated Recurrent Unit) model outperformed LSTM for train presence "
                "detection, achieving a 94% detection rate compared to 93% for LSTM. Both "
                "models used the average trend of TDMS signals to extract train presence or "
                "absence conditions. The GRU architecture is preferred for DAS applications "
                "due to its fewer parameters and faster convergence on long time-series data.",

                "The average trend extraction method was applied to the large and noisy DAS "
                "dataset to reduce dimensionality while preserving condition-relevant patterns. "
                "This preprocessing step computes a rolling mean across the multi-channel "
                "signal matrix, smoothing transient noise while retaining sustained acoustic "
                "events from train passage. This proved more effective than raw signal "
                "processing for both GRU and LSTM training.",
            ]
        },

        # ── Paper 3: DAS Review ───────────────────────────────────────────────
        {
            "title": "A review of distributed acoustic sensing applications for "
                     "railroad condition monitoring",
            "authors": "Rahman MA et al.",
            "year": "2023",
            "doi": "10.1016/j.ymssp.2023.110881",
            "venue": "Mechanical Systems and Signal Processing, Elsevier",
            "domain": "das_review",
            "chunks": [
                "DAS technology enables continuous, real-time monitoring of railroad "
                "infrastructure by converting a standard optical fiber cable into an array "
                "of thousands of virtual microphones. The key advantages over traditional "
                "point sensors include: (1) distributed sensing over tens of kilometers from "
                "a single interrogator unit, (2) no electrical power required along the "
                "sensing fiber, (3) immunity to electromagnetic interference, and (4) "
                "simultaneous monitoring of multiple train events.",

                "Current challenges in DAS-based railroad monitoring include: large data "
                "volumes (terabytes per day for long rail networks), high noise levels from "
                "environmental sources (wind, temperature variations, road traffic), class "
                "imbalance between normal and anomalous conditions, and the need for "
                "real-time processing algorithms that can operate within operational latency "
                "constraints. Deep learning approaches, particularly CNN-LSTM hybrids, "
                "have shown the most promise for addressing these challenges.",

                "Signal processing approaches for DAS data analysis span three domains: "
                "conventional signal processing (peak detection, filtering, wavelet transforms), "
                "classical machine learning (SVM, KNN, Random Forest on extracted features), "
                "and deep learning (CNN, LSTM, GRU, hybrid architectures). Deep learning "
                "methods consistently outperform classical approaches on large-scale DAS "
                "datasets due to their ability to automatically learn multi-scale temporal "
                "and spectral features without manual feature engineering.",
            ]
        },

        # ── Paper 4: WAAM HMM-RL ──────────────────────────────────────────────
        {
            "title": "Hidden Markov Chain-Reinforcement Learning (HMM-RL) for "
                     "WAAM intelligent control",
            "authors": "Rahman MA et al.",
            "year": "2026",
            "doi": "Under review — Springer",
            "venue": "Springer 2026",
            "domain": "manufacturing_ai",
            "chunks": [
                "Wire Arc Additive Manufacturing (WAAM) is a directed energy deposition "
                "process that builds metallic parts layer by layer using an electric arc as "
                "the heat source. Key process parameters include wire feed speed, travel "
                "speed, current, and shielding gas flow. Material properties (porosity, "
                "hardness, tensile strength) are highly sensitive to these parameters, "
                "making real-time intelligent control essential for consistent part quality.",

                "The HMM-RL pipeline models WAAM process states using a Hidden Markov Model "
                "(HMM) to capture the latent manufacturing states from sensor observations, "
                "and uses Reinforcement Learning (RL) to optimize the control policy across "
                "these states. The HMM identifies transitions between process regimes "
                "(stable deposition, thermal buildup, cooling) from temperature and current "
                "sensor data. The RL agent then learns to adjust process parameters to "
                "maximize material quality while minimizing waste. This approach improved "
                "material utilization by 5% on live WAAM datasets.",

                "The Georgia-AIM grant supports development of AI-driven manufacturing "
                "prototypes under the Partnership for Innovation (PIN) program at Georgia "
                "Tech. The WAAM optimization work is part of a broader initiative to "
                "integrate ML-based process control into American manufacturing facilities, "
                "targeting improved efficiency, reduced scrap rates, and enhanced part "
                "quality for aerospace and defense applications.",
            ]
        },

    ]

    chunks = []
    for paper in papers:
        meta = {k: v for k, v in paper.items() if k != "chunks"}
        for chunk_text in paper["chunks"]:
            chunks.append(DocumentChunk(text=chunk_text, metadata=meta))

    return chunks
