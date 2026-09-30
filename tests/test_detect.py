from detect import detect, make_sensor_data


def test_detector_finds_injected_faults():
    data = make_sensor_data()
    _, _, metrics, scored = detect(data)
    assert len(scored) == len(data)
    assert scored["predicted_anomaly"].sum() > 0
    assert metrics["f1"] > 0.7
