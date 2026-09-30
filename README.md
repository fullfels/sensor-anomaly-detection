# Sensor Anomaly Detection

An unsupervised machine-learning example that detects abnormal sensor behavior with Isolation Forest. The script generates multivariate telemetry, injects known faults, scores every observation, and creates a diagnostic plot.

## Highlights

- Multivariate synthetic telemetry with realistic correlations
- Isolation Forest for unsupervised detection
- Precision, recall, and F1 against injected ground truth
- Ranked anomaly scores and visualization
- Reproducible command-line workflow

## Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
python detect.py --output-dir artifacts
pytest -q
```

## Outputs

- `artifacts/scored_sensor_data.csv`
- `artifacts/metrics.json`
- `artifacts/anomalies.png`

## License

MIT
