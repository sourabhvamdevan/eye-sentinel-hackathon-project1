

# EYESENTINEL — Assumptions & Limitations

## Assumptions

1. **Dataset validity**: The Kaggle "Fondo de Ojo" fundus image dataset used for training is assumed to be correctly labeled and representative of real-world retinal presentations.
2. **Image quality**: User-uploaded fundus images are assumed to be reasonably clear, correctly oriented, and captured with standard fundus photography equipment (not phone photos of a screen, etc.).
3. **Single-eye, single-condition framing**: Each uploaded image is assumed to represent one eye at a time, and the model outputs are assumed to be interpreted per-image rather than per-patient.
4. **Environment consistency**: The `models/eye_disease_model.h5` weights are assumed to match the architecture defined in the app/config code — i.e., the model was trained and exported using compatible TensorFlow/Keras versions.
5. **Local/cloud parity**: Behavior on Render (or other cloud deployment) is assumed to mirror local `streamlit run app.py` behavior, assuming environment variables and dependency versions are kept in sync via `requirements.txt`.
6. **Non-clinical usage context**: It is assumed that a qualified clinician or trained user reviews AI outputs before any decision-making — the tool is assumed to be used as a triage aid, not a standalone diagnostic authority.
7. **Static thresholding**: Sigmoid probability thresholds for classifying risk (e.g., glaucoma positive/negative) are assumed to be fixed at reasonable defaults set during development, not dynamically calibrated per population.

## Limitations

1. **Not FDA/CE approved**: EYESENTINEL is a hackathon/prototype-grade project and is **not** a certified medical device. It must not be used for real clinical diagnosis or treatment decisions.
2. **Dataset bias**: The model is only as good as the Kaggle dataset it was trained on — it may not generalize well across different ethnicities, camera hardware, lighting conditions, or disease severities not represented in training data.
3. **Limited disease scope**: Only screens for Glaucoma, Diabetic Retinopathy, Cataract, and "normal" baseline — it cannot detect other ocular or systemic conditions (e.g., macular degeneration, retinal detachment, tumors).
4. **No patient history integration**: The system evaluates images in isolation; it does not incorporate patient history, IOP (intraocular pressure) readings, visual field tests, or other clinical context that a real diagnosis would require.
5. **Grad-CAM is illustrative, not definitive**: Heatmaps show where the model "looked," not necessarily clinically meaningful pathology — they can be misleading if the model has learned spurious correlations.
6. **Single-model architecture**: Relies on one custom CNN rather than an ensemble, so performance may be less robust than production-grade multi-model clinical systems.
7. **No real-time model retraining**: The app serves a static, pre-trained `.h5` model; it does not learn or improve from new uploads during runtime.
8. **Report generation limitations**: FPDF2-generated PDF reports are formatted outputs of model predictions and are not legally or medically binding documents.
9. **Scalability constraints**: Streamlit + Render deployment is suited for demos/prototypes, not high-throughput clinical environments with many concurrent users.
10. **Data privacy**: No explicit mention of HIPAA/GDPR-compliant handling of uploaded medical images — uploaded images should be treated as sensitive data, and production use would require proper compliance review.