"""Detect anomalies in multivariate sensor telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler


FEATURES = ["temperature", "pressure", "vibration"]


def make_sensor_data(n_samples: int = 1800, anomaly_fraction: float = 0.04, random_state: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    time = pd.date_range("2026-01-01", periods=n_samples, freq="min")
    cycle = np.sin(np.arange(n_samples) * 2 * np.pi / 240)
    temperature = 42 + 3.2 * cycle + rng.normal(0, 0.65, n_samples)
    pressure = 101.5 + 0.45 * cycle + 0.09 * (temperature - 42) + rng.normal(0, 0.18, n_samples)
    vibration = 0.85 + 0.08 * cycle + rng.normal(0, 0.035, n_samples)
    truth = np.zeros(n_samples, dtype=int)
    anomaly_count = int(n_samples * anomaly_fraction)
    indices = rng.choice(np.arange(30, n_samples - 30), anomaly_count, replace=False)
    truth[indices] = 1
    temperature[indices] += rng.choice([-1, 1], anomaly_count) * rng.uniform(5, 10, anomaly_count)
    pressure[indices] += rng.choice([-1, 1], anomaly_count) * rng.uniform(0.8, 1.8, anomaly_count)
    vibration[indices] += rng.uniform(0.22, 0.55, anomaly_count)
    return pd.DataFrame(
        {"timestamp": time, "temperature": temperature, "pressure": pressure, "vibration": vibration, "is_anomaly": truth}
    )


def detect(frame: pd.DataFrame, contamination: float = 0.04, random_state: int = 21):
    scaler = StandardScaler()
    X = scaler.fit_transform(frame[FEATURES])
    model = IsolationForest(n_estimators=250, contamination=contamination, random_state=random_state, n_jobs=-1)
    predicted = (model.fit_predict(X) == -1).astype(int)
    scored = frame.copy()
    scored["anomaly_score"] = -model.score_samples(X)
    scored["predicted_anomaly"] = predicted
    precision, recall, f1, _ = precision_recall_fscore_support(
        scored["is_anomaly"], predicted, average="binary", zero_division=0
    )
    metrics = {"precision": round(float(precision), 3), "recall": round(float(recall), 3), "f1": round(float(f1), 3)}
    return model, scaler, metrics, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, _, metrics, scored = detect(make_sensor_data())
    scored.to_csv(args.output_dir / "scored_sensor_data.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    flagged = scored[scored["predicted_anomaly"] == 1]
    plt.figure(figsize=(11, 4))
    plt.plot(scored["timestamp"], scored["temperature"], linewidth=0.9, label="temperature")
    plt.scatter(flagged["timestamp"], flagged["temperature"], color="crimson", s=18, label="detected anomaly")
    plt.ylabel("temperature")
    plt.title("Sensor anomaly detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "anomalies.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
