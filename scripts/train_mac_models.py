#!/usr/bin/env python3
"""
Train XGBoost AFib Classifier + Isolation Forest Anomaly Detector
===================================================================
Runs on Mac CPU in ~2 minutes using real AFDB data.
Produces two models that work immediately in the dashboard.
"""

import os, sys, json, logging, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wfdb
from heartengine.stage_a.pan_tompkins import AdaptivePanTompkins
from heartengine.stage_a.hrv_features import extract_hrv_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MacTrainer")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "afdb")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW_BEATS = 60  # analyze 60-beat windows


def extract_windows_from_record(rec_path):
    """Extract HRV feature windows with AF/Normal labels from one AFDB record."""
    try:
        rec = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")
    except Exception as e:
        logger.warning(f"Skip {rec_path}: {e}")
        return [], []

    fs = rec.fs
    MAX_DURATION_SEC = 30 * 60
    max_samples = MAX_DURATION_SEC * fs
    signal = rec.p_signal[:max_samples, 0].astype(np.float64)

    # Get rhythm spans, adjust for cropped signal
    rhythm_spans = []
    for i, (samp, sym, aux) in enumerate(zip(ann.sample, ann.symbol, ann.aux_note)):
        if samp >= max_samples:
            break
        if sym == "+" and aux.strip():
            end = max_samples
            for j in range(i + 1, len(ann.sample)):
                if ann.sample[j] >= max_samples:
                    end = max_samples
                    break
                if ann.symbol[j] == "+":
                    end = ann.sample[j]
                    break
            rhythm_spans.append((samp, end, aux.strip()))

    # Detect R-peaks
    pt = AdaptivePanTompkins()
    result = pt.detect(signal, fs)
    rpeaks = result.rpeaks
    rr = result.rr_intervals_sec

    if len(rr) < WINDOW_BEATS + 10:
        return [], []

    # Map each R-peak to its rhythm label
    peak_labels = []
    for pk in rpeaks:
        label = "N"
        for start, end, rtype in rhythm_spans:
            if start <= pk < end:
                rtype_clean = rtype.strip("()")
                if rtype_clean in ("AFIB", "AFL"):
                    label = "AF"
                break
        peak_labels.append(label)

    # Slide windows of WINDOW_BEATS
    features_list = []
    labels_list = []

    for i in range(0, len(rr) - WINDOW_BEATS, WINDOW_BEATS // 2):
        window_rr = rr[i:i + WINDOW_BEATS]
        window_labels = peak_labels[i:i + WINDOW_BEATS]

        # Label: majority vote
        af_count = sum(1 for l in window_labels if l == "AF")
        is_af = 1 if af_count > WINDOW_BEATS * 0.5 else 0

        # Extract HRV features
        try:
            hrv = extract_hrv_features(window_rr)
            if hrv and len(hrv) > 5:
                features_list.append(hrv)
                labels_list.append(is_af)
        except:
            pass

    return features_list, labels_list


def train():
    logger.info("=" * 60)
    logger.info("Training XGBoost + Isolation Forest on AFDB (Mac CPU)")
    logger.info("=" * 60)

    # Collect data from all available records
    records = sorted([f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".dat")])
    logger.info(f"Found {len(records)} AFDB records")

    all_features = []
    all_labels = []

    for rec_name in records:
        logger.info(f"Processing {rec_name}...")
        feats, labels = extract_windows_from_record(os.path.join(DATA_DIR, rec_name))
        all_features.extend(feats)
        all_labels.extend(labels)
        logger.info(f"  → {len(feats)} windows (AF={sum(labels)}, Normal={len(labels)-sum(labels)})")

    if len(all_features) < 10:
        logger.error("Not enough data. Need more AFDB records.")
        return

    # Convert to matrix
    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(k, 0) for k in feature_names] for f in all_features])
    y = np.array(all_labels)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    logger.info(f"\nDataset: {X.shape[0]} windows, {X.shape[1]} features")
    logger.info(f"  AF: {sum(y)}, Normal: {len(y) - sum(y)}")

    # ========== XGBOOST ==========
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    try:
        import xgboost as xgb
        logger.info("\nTraining XGBoost...")

        # Handle imbalance
        n_pos = sum(y_train)
        n_neg = len(y_train) - n_pos
        scale_pos = n_neg / n_pos if n_pos > 0 else 1

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        logger.info(f"XGBoost F1: {f1:.4f}")
        logger.info(classification_report(y_test, y_pred, target_names=["Normal", "AFib"]))

        # Save
        xgb_path = os.path.join(MODEL_DIR, "xgboost_afib.pkl")
        with open(xgb_path, "wb") as f:
            pickle.dump({"model": model, "feature_names": feature_names}, f)
        logger.info(f"✓ Saved XGBoost → {xgb_path}")

    except ImportError:
        logger.warning("XGBoost not installed, using RandomForest fallback")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        logger.info(f"RandomForest F1: {f1:.4f}")
        rf_path = os.path.join(MODEL_DIR, "xgboost_afib.pkl")
        with open(rf_path, "wb") as f:
            pickle.dump({"model": model, "feature_names": feature_names}, f)

    # ========== ISOLATION FOREST (Novel: Unsupervised Anomaly Detection) ==========
    from sklearn.ensemble import IsolationForest

    logger.info("\nTraining Isolation Forest (unsupervised anomaly detector)...")

    # Train ONLY on normal rhythm — anomalies = anything that deviates
    X_normal = X[y == 0]
    logger.info(f"  Training on {len(X_normal)} normal windows")

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.1,  # expect ~10% anomalies
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_normal)

    # Evaluate: anomaly score on test set
    scores = iso_forest.decision_function(X_test)
    preds_iso = (iso_forest.predict(X_test) == -1).astype(int)  # -1 = anomaly
    f1_iso = f1_score(y_test, preds_iso)
    logger.info(f"Isolation Forest F1: {f1_iso:.4f}")

    iso_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    with open(iso_path, "wb") as f:
        pickle.dump({"model": iso_forest, "feature_names": feature_names}, f)
    logger.info(f"✓ Saved Isolation Forest → {iso_path}")

    # ========== SUMMARY ==========
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"  XGBoost AFib F1:        {f1:.4f}")
    logger.info(f"  Isolation Forest F1:    {f1_iso:.4f}")
    logger.info(f"  Models saved to:        {MODEL_DIR}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    train()
