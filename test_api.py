# ==========================================================================================
# AUTOMATED API VERIFICATION TEST SUITE
# ==========================================================================================

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def run_tests():
    print("=" * 80)
    print("STARTING API TEST SUITE")
    print("=" * 80)

    # Test 1: Health
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    assert health_data["status"] == "ok"
    assert health_data["models"]["v4_tech_classifier"] is True
    assert health_data["models"]["v4_salary_regressor"] is True
    print("[PASS] 1. Health check passed.")

    # Test 2: Standard 7-Feature Prediction
    p1 = client.post("/predict", json={
        "cgpa": 8.4,
        "aptitude_score": 82.0,
        "internships": 2,
        "projects_count": 3,
        "certifications": 2,
        "communication_skills": 7.0,
        "extracurricular_activities": 6.0,
    })
    assert p1.status_code == 200, f"Standard predict failed: {p1.text}"
    data1 = p1.json()
    assert data1["status"] in ["Placed", "Not Placed"]
    assert 0.0 <= data1["probability"] <= 1.0
    assert "prediction_id" in data1
    assert "shap_factors" in data1
    assert len(data1["top_positive_factors"]) > 0 or len(data1["top_negative_factors"]) > 0
    if data1["status"] == "Placed":
        assert data1["expected_salary_lpa"] is not None
        assert data1["salary_range"] is not None
        print(f"[PASS] 2. Standard predict passed: {data1['status']} ({data1['probability'] * 100:.1f}%), Salary: {data1['expected_salary_lpa']} LPA")
    else:
        print(f"[PASS] 2. Standard predict passed: {data1['status']} ({data1['probability'] * 100:.1f}%)")

    # Test 3: Technical-Extended Prediction (V4 Features)
    p2 = client.post("/predict", json={
        "cgpa": 6.2,
        "aptitude_score": 50.0,
        "internships": 0,
        "projects_count": 1,
        "certifications": 0,
        "communication_skills": 4.0,
        "extracurricular_activities": 2.0,
        "coding_skills": 3.0,
        "dsa_score": 2.5,
        "backlogs": 2,
        "college_tier": "Tier-3",
        "branch": "ME",
    })
    assert p2.status_code == 200, f"Tech predict failed: {p2.text}"
    data2 = p2.json()
    assert data2["status"] == "Not Placed"
    assert data2["risk_level"] == "High Risk"
    assert len(data2["top_negative_factors"]) > 0
    print(f"[PASS] 3. High-risk tech predict passed: {data2['status']} ({data2['probability'] * 100:.1f}%), Top Risk: {data2['top_negative_factors'][0]}")

    # Test 4: Model Metrics
    m = client.get("/model/metrics")
    assert m.status_code == 200, f"Metrics failed: {m.text}"
    m_data = m.json()
    assert "internal" in m_data
    assert "comparison" in m_data
    print(f"[PASS] 4. Model metrics endpoint passed: {m_data['model_name']}")

    # Test 5: Dashboard Statistics
    d = client.get("/dashboard/statistics")
    assert d.status_code == 200, f"Dashboard failed: {d.text}"
    d_data = d.json()
    assert "placement_rate" in d_data
    assert "probability_distribution" in d_data
    print(f"[PASS] 5. Dashboard statistics passed: Total students = {d_data['total_students']}, Rate = {d_data['placement_rate']}")

    # Test 6: Analytics Data
    a = client.get("/analytics")
    assert a.status_code == 200, f"Analytics failed: {a.text}"
    a_data = a.json()
    assert "probability_by_cgpa" in a_data
    print("[PASS] 6. Analytics endpoint passed.")

    # Test 7: History Persistence (Save and Delete)
    test_record = {
        "prediction_id": "PR-TEST-999",
        "timestamp": "2026-09-10T14:30:00Z",
        "saved": True,
        "cgpa": 8.0,
        "aptitude_score": 80.0,
        "internships": 2,
        "projects_count": 3,
        "certifications": 1,
        "communication_skills": 7.0,
        "extracurricular_activities": 5.0,
        "status": "Placed",
        "prediction": 1,
        "probability": 0.88,
        "risk_level": "Low Risk",
        "expected_salary_lpa": 18.5,
        "salary_range": "17.5 - 19.5 LPA",
    }
    s = client.post("/predictions", json=test_record)
    assert s.status_code == 200, f"Save prediction failed: {s.text}"

    h = client.get("/predictions")
    assert h.status_code == 200
    records = h.json()
    assert any(r["prediction_id"] == "PR-TEST-999" for r in records)
    print("[PASS] 7. Save prediction passed.")

    del_res = client.delete("/predictions/PR-TEST-999")
    assert del_res.status_code == 200
    h_after = client.get("/predictions")
    assert not any(r["prediction_id"] == "PR-TEST-999" for r in h_after.json())
    print("[PASS] 8. Delete prediction passed.")

    print("=" * 80)
    print("ALL 8 TEST SUITES COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
