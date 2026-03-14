"""
Built-in Knowledge Base — Md Arifur Rahman
Comprehensive knowledge covering:
- Full work experience (2014-present)
- Education background
- All 7 published papers
- Certifications & professional development
- Skills & technical expertise
"""

from src.retrieval.rag_pipeline import DocumentChunk


def get_builtin_chunks() -> list[DocumentChunk]:

    entries = [

        # ── PROFESSIONAL PROFILE ──────────────────────────────────────────────
        {
            "title": "Professional Profile — Md Arifur Rahman",
            "category": "profile",
            "text": (
                "Md Arifur Rahman is an innovative ML Engineer and AI Researcher based in "
                "Atlanta, GA with strong foundations in analytics, software development, "
                "machine learning, and data science. He applies advanced research to "
                "enterprise use cases, integrating generative AI, predictive maintenance, "
                "and real-time dashboards into cloud-native model pipelines. He builds "
                "scalable full-stack solutions and deploys ML models that optimize processes "
                "and enable intelligent decision-making. He is authorized to work in the US "
                "with no sponsorship required. Contact: arifme071@gmail.com | 912-541-9169 | "
                "Atlanta, GA 30349."
            )
        },

        # ── WORK EXPERIENCE ───────────────────────────────────────────────────
        {
            "title": "Current Role: PIN Fellow — AI in Manufacturing, Georgia Tech (2025–Present)",
            "category": "experience",
            "text": (
                "Md Arifur Rahman is currently a PIN Fellow (AI in Manufacturing) at the "
                "Partnership for Innovation at Georgia Tech, Atlanta GA, starting July 2025. "
                "This is under the Georgia-AIM grant to advance manufacturing optimization. "
                "He developed ML prototypes including hidden Markov model and reinforcement "
                "learning pipelines on WAAM datasets. He engineered AI-driven tools to align "
                "production with dynamic demand/supply signals and established MLOps best "
                "practices across cross-functional teams. Key achievements: improved material "
                "utilization by 5% using HMM-RL on WAAM datasets; deployed an AI-driven "
                "material design prototype to production; authored findings in a peer-reviewed "
                "Springer publication (2026)."
            )
        },
        {
            "title": "Previous Role: Data Analyst at Norfolk Southern Corporation (2024–2025)",
            "category": "experience",
            "text": (
                "At Norfolk Southern Corporation in Atlanta GA (January 2024 – March 2025), "
                "Arifur worked as a Data Analyst/Reporting Supervisor Associate on the Digital "
                "Train Inspection team. He delivered end-to-end solutions from server-side data "
                "processing to automated reporting for zone-wise inspection leaders and senior "
                "stakeholders. He built GIS-enabled Power BI and Tableau dashboards to visualize "
                "train health and inspection data. Key achievements: automated workflows using "
                "Alteryx/SQL reducing manual processing by 3%; developed GIS dashboards for "
                "real-time train health monitoring adopted company-wide; identified inspection "
                "inefficiencies improving turnaround time by 5%."
            )
        },
        {
            "title": "Research Role: ML & Data Analytics Research Assistant, Georgia Southern University (2022–2023)",
            "category": "experience",
            "text": (
                "At Georgia Southern University in Statesboro GA (August 2022 – December 2023), "
                "Arifur worked as a Research Assistant in ML & Data Analytics. He conducted "
                "research on rail systems safety and reliability, developing anomaly detection "
                "models and real-time monitoring tools on live industry datasets. Key achievements: "
                "built ML models for anomaly detection on live HTL loop-MxV rail datasets from "
                "AAR/TTCI Pueblo CO achieving 95% accuracy; published 7 peer-reviewed papers in "
                "ASME and Springer journals accumulating 110+ citations; developed Python/SQL "
                "dashboards for real-time asset monitoring; led data collection under AAR/TTCI "
                "and Georgia Southern University grants."
            )
        },
        {
            "title": "Bangladesh Experience: Deputy Manager at Aramit Limited (2018–2022)",
            "category": "experience",
            "text": (
                "Before coming to the US, Arifur served as Deputy Manager at Aramit Limited "
                "in Bangladesh from February 2018 to July 2022 — nearly 4.5 years. He "
                "demonstrated expertise in mechanical and process development, overseeing "
                "mechanical issues and leading preventive maintenance initiatives. He managed "
                "production equipment, hydraulic and pneumatic systems, and cross-functional "
                "engineering teams. This role gave him 4+ years of hands-on manufacturing "
                "and operations management experience."
            )
        },
        {
            "title": "Bangladesh Experience: Assistant Engineer at KSRM Billet Industries (2014–2018)",
            "category": "experience",
            "text": (
                "Arifur began his professional career as an Assistant Engineer at KSRM Billet "
                "Industries Ltd in Bangladesh, a steel manufacturing company. From 2014 to 2018 "
                "he gained foundational experience in mechanical engineering, equipment "
                "maintenance, production operations, and industrial process management. "
                "This role combined with his Aramit Limited experience gives him over "
                "8 years of total professional experience spanning manufacturing, operations, "
                "data analytics, and AI/ML research."
            )
        },

        # ── EDUCATION ─────────────────────────────────────────────────────────
        {
            "title": "Education: Admitted to Georgia Tech OMSCS (MS Computer Science, Fall 2026)",
            "category": "education",
            "text": (
                "Arifur has been admitted to the Georgia Institute of Technology's Online "
                "Master of Science in Computer Science (OMSCS) program, starting Fall 2026. "
                "This is a part-time online program at one of the top CS programs in the US. "
                "Georgia Tech OMSCS is ranked among the best value CS master's programs "
                "globally. This admission reflects his strong academic and research background."
            )
        },
        {
            "title": "Education: MSc Applied Engineering, Georgia Southern University",
            "category": "education",
            "text": (
                "Arifur earned his MSc in Applied Engineering with a specialization in "
                "Advanced Manufacturing Engineering from Georgia Southern University, "
                "Statesboro GA. His thesis focused on acoustic sensing for railroad "
                "predictive maintenance. During his MSc he was a Graduate Research Assistant "
                "at the LANDTIE Research Lab under Professor Hossein Taheri. He was awarded "
                "the Gene Haas Foundation Manufacturing Engineering Scholarship for 2022-2023. "
                "His MSc research produced 7 peer-reviewed publications and 184+ citations."
            )
        },
        {
            "title": "Education: BSc Mechanical Engineering, CUET Bangladesh (2014)",
            "category": "education",
            "text": (
                "Arifur earned his Bachelor of Science in Mechanical Engineering from "
                "Chittagong University of Engineering and Technology (CUET) in Bangladesh "
                "in 2014, graduating with a merit position of 35 out of 127 students. "
                "CUET is one of Bangladesh's premier engineering universities. His BS "
                "foundation in mechanical engineering underpins his later expertise in "
                "manufacturing systems, structural health monitoring, and industrial AI."
            )
        },

        # ── PAPERS 1-4 (existing) ─────────────────────────────────────────────
        {
            "title": "Paper 1: CNN-LSTM-SW for Railroad Anomaly Detection via DAS (Elsevier 2024)",
            "category": "research",
            "text": (
                "Published in Green Energy and Intelligent Transportation, Elsevier 2024. "
                "DOI: 10.1016/j.geits.2024.100178. Authors: Rahman MA, Jamal S, Taheri H. "
                "The CNN-LSTM-SW model combines CNNs for spatial feature extraction, LSTMs "
                "for temporal learning, and a sliding window for misclassification correction. "
                "Validated on 4.16 km HTL loop DAS data from AAR/TTCI Pueblo CO. "
                "Results: 97% train position detection, ROC AUC AC1=1.00, NC=0.99, TP=0.97, AC2=0.95. "
                "Four classes: NC (Normal), TP (Train Position), AC1, AC2. "
                "SMOTE handles class imbalance. Combined T&FD features outperform TD-only and FD-only."
            )
        },
        {
            "title": "Paper 2: DAS Railroad CM with GRU/LSTM (SPIE 2024)",
            "category": "research",
            "text": (
                "Published in Journal of Applied Remote Sensing, SPIE 2024. "
                "DOI: 10.1117/1.JRS.18.016512. Authors: Rahman MA, Kim J, Dababneh F, Taheri H. "
                "GRU model achieved 94% vs LSTM 93% for train presence detection using DAS signals. "
                "The iDAS system (Silixa) has gauge length 10m, sensing range 45km, up to 100kHz sampling. "
                "Average trend extraction preprocesses large noisy DAS data via rolling mean. "
                "DAS uses Rayleigh backscattering in fiber optic cables as distributed sensors."
            )
        },
        {
            "title": "Paper 3: Review of DAS Applications for Railroad CM (Elsevier MSSP 2023)",
            "category": "research",
            "text": (
                "Published in Mechanical Systems and Signal Processing, Elsevier 2023. "
                "DOI: 10.1016/j.ymssp.2023.110881. Systematic review of distributed acoustic "
                "sensing applications for railroad condition monitoring. DAS challenges: "
                "terabytes of daily data, high environmental noise, class imbalance, real-time constraints. "
                "Deep learning (CNN, LSTM, GRU hybrids) consistently outperforms classical ML "
                "(SVM, KNN, Random Forest) on large DAS datasets. Widely cited review paper."
            )
        },
        {
            "title": "Paper 4: HMM-RL for WAAM Manufacturing (Springer 2026)",
            "category": "research",
            "text": (
                "Under review at Springer 2026. Authors: Rahman MA et al. "
                "HMM-RL pipeline for Wire Arc Additive Manufacturing: Hidden Markov Model "
                "captures latent process states (stable deposition, thermal buildup, cooling) "
                "from temperature and current sensors. RL agent optimizes wire feed speed, "
                "travel speed, current, and gas flow to maximize material quality. "
                "Achieved 5% material utilization improvement on live WAAM datasets "
                "under Georgia-AIM grant at Georgia Tech."
            )
        },

        # ── PAPERS 5-7 (new) ──────────────────────────────────────────────────
        {
            "title": "Paper 5: AI-Guided Optimization of Polymer Film Synthesis (Springer 2026)",
            "category": "research",
            "text": (
                "Under review at Springer 2026. Authors: Rahman MA et al. "
                "AI-guided optimization of polymer film synthesis using ML surrogates and "
                "multi-objective design. Applies machine learning surrogate models to replace "
                "expensive physical simulations in polymer manufacturing. Multi-objective "
                "optimization balances competing material properties (strength, flexibility, "
                "thermal stability). Part of the broader Georgia-AIM manufacturing AI initiative "
                "at Georgia Tech PIN program."
            )
        },
        {
            "title": "Paper 6: Structural Health Monitoring for Railroad Infrastructure",
            "category": "research",
            "text": (
                "Published in ASME journal. Authors: Rahman MA, Taheri H et al. "
                "Structural health monitoring (SHM) for railroad infrastructure using "
                "sensor data and machine learning. Focuses on early detection of structural "
                "degradation in rail tracks, bridges, and related infrastructure. "
                "Combines vibration analysis, acoustic sensing, and ML-based pattern "
                "recognition for continuous automated monitoring. Part of the broader "
                "body of work from the LANDTIE Research Lab at Georgia Southern University."
            )
        },
        {
            "title": "Paper 7: Predictive Maintenance for Rail Systems Using ML",
            "category": "research",
            "text": (
                "Published in ASME/Springer. Authors: Rahman MA, Taheri H et al. "
                "Machine learning approaches for predictive maintenance of rail systems. "
                "Covers anomaly detection on live HTL loop-MxV rail datasets from AAR/TTCI "
                "achieving 95% accuracy with reduced false inspection alarms. "
                "Developed real-time Python/SQL monitoring dashboards adopted by faculty "
                "and industry partners. Led data collection under AAR/TTCI and Georgia "
                "Southern University research grants."
            )
        },

        # ── CERTIFICATIONS ────────────────────────────────────────────────────
        {
            "title": "Certifications & Professional Development",
            "category": "certifications",
            "text": (
                "Arifur holds the following certifications: "
                "Google Cloud Data Analytics Certificate; "
                "Alteryx Designer Core Certification; "
                "Coursera Applied Machine Learning in Python (University of Michigan). "
                "Professional development includes: Oracle SQL Explorer; "
                "Udemy LLM Engineering; Udemy Google Cloud Professional Data Engineer; "
                "LinkedIn Learning Power BI; Udemy ML for Industry 4.0; "
                "Udemy Data Analyst Bootcamp; Google Generative AI Leader. "
                "These certifications span cloud platforms, ML engineering, data analytics, "
                "and generative AI — directly aligned with FAANG technical requirements."
            )
        },

        # ── TECHNICAL SKILLS ──────────────────────────────────────────────────
        {
            "title": "Technical Skills — Languages & ML/AI",
            "category": "skills",
            "text": (
                "Programming languages: Python (primary), SQL, Java, MATLAB, C++. "
                "ML & AI frameworks: LLMs (GPT, BERT), TensorFlow, Hugging Face, "
                "Vertex AI, RAG pipelines, AutoML, LangChain, LangGraph. "
                "Data tools: Pandas, scikit-learn, Alteryx, NumPy, SciPy. "
                "Software engineering: Git, REST APIs, Agile methodology, Streamlit, Docker. "
                "Specialized ML: CNN, LSTM, GRU, Hidden Markov Models, Reinforcement Learning, "
                "Anomaly Detection, Predictive Maintenance, Time-Series Analysis, FAISS, "
                "SentenceTransformers, Signal Processing (FFT, STFT, wavelets)."
            )
        },
        {
            "title": "Technical Skills — Cloud, Visualization & Engineering Tools",
            "category": "skills",
            "text": (
                "Cloud platforms: Google Cloud (BigQuery, Vertex AI, Cloud Functions), "
                "Oracle, AWS (S3, Lambda), Azure (VM). "
                "Data visualization: Power BI, Looker, Tableau, GIS mapping, Streamlit. "
                "Engineering software: AutoCAD, SolidWorks, PLC diagnostics, Ansys simulation. "
                "Areas of expertise: Machine Learning & AI, Reinforcement Learning, "
                "Generative AI & LLMs, Predictive Analytics & Anomaly Detection, "
                "MLOps & Model Deployment, Data Pipeline Automation, "
                "PLC Diagnostics & Troubleshooting, Asset & Structural Health Monitoring, "
                "Manufacturing & Transportation Optimization, Cross-Functional Collaboration."
            )
        },

        # ── AREAS OF EXPERTISE ────────────────────────────────────────────────
        {
            "title": "Target Roles & Career Goals",
            "category": "profile",
            "text": (
                "Arifur is open to roles in ML Engineering, AI/Data Science, Data Engineering, "
                "and AI Research — particularly in manufacturing, transportation, energy, "
                "and infrastructure intelligence. Target companies include Google, Microsoft, "
                "Intel, and other major tech companies. He brings a unique combination of "
                "deep domain expertise (railroad AI, manufacturing optimization) with "
                "production ML engineering skills (RAG pipelines, CNN-LSTM, MLOps). "
                "He is US work authorized with no sponsorship required, based in Atlanta GA. "
                "Starting Georgia Tech OMSCS (MS Computer Science) Fall 2026."
            )
        },
    ]

    chunks = []
    for entry in entries:
        meta = {
            "title": entry["title"],
            "authors": "Md Arifur Rahman",
            "year": "2024",
            "source": entry["category"],
            "domain": entry["category"],
            "venue": entry.get("venue", ""),
        }
        chunks.append(DocumentChunk(text=entry["text"], metadata=meta))

    return chunks
