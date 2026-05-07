"""
pytest test suite for the expert system pipeline.
Run: pytest tests/ -v
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'interface'))
from predictor import diagnose_patient, combine_cf, CF_RULES

def test_known_high_risk_patient():
    """Classic preeclampsia profile — should be HIGH risk."""
    patient = {
        'Age': 38, 'SystolicBP': 158, 'DiastolicBP': 102,
        'BS': 15.0, 'BodyTemp': 101.0, 'HeartRate': 96
    }
    result = diagnose_patient(patient)
    assert 'HIGH' in result['verdict']


def test_known_low_risk_patient():
    """Young healthy patient — should be LOW risk."""
    patient = {
        'Age': 22, 'SystolicBP': 112, 'DiastolicBP': 72,
        'BS': 5.5, 'BodyTemp': 98.2, 'HeartRate': 74
    }
    result = diagnose_patient(patient)
    assert 'LOW' in result['verdict']


def test_fever_threshold_boundary():
    """Patient at exactly 100.4°F — fever rule should fire."""
    patient = {
        'Age': 25, 'SystolicBP': 118, 'DiastolicBP': 76,
        'BS': 6.0, 'BodyTemp': 100.4, 'HeartRate': 78
    }
    result = diagnose_patient(patient)
    label = result['verdict']
    assert label in ['🔴 Likely HIGH risk', '🟡 Possibly MID risk', '⚪ Uncertain — borderline', '🟢 Likely LOW risk']
    # Fever should fire
    assert 'fever' in result['rules_fired']


def test_cf_combination_positive():
    """Two positive CFs combine correctly."""
    result = combine_cf(0.70, 0.50)
    expected = 0.70 + 0.50 * (1 - 0.70)   # = 0.85
    assert abs(result - expected) < 1e-9


def test_cf_combination_negative():
    """Two negative CFs combine correctly."""
    result = combine_cf(-0.40, -0.25)
    expected = -0.40 + (-0.25) * (1 + (-0.40))   # = -0.55
    assert abs(result - expected) < 1e-9


def test_cf_combination_mixed():
    """Positive and negative CFs combine correctly."""
    result = combine_cf(0.70, -0.40)
    expected = (0.70 + (-0.40)) / (1 - min(abs(0.70), abs(-0.40)))
    assert abs(result - expected) < 1e-9


def test_all_rules_defined():
    """All CF rules must have required fields."""
    for rule in CF_RULES:
        assert 'name' in rule
        assert 'cf' in rule
        assert 'condition' in rule
        assert 'label' in rule
        assert callable(rule['condition'])


def test_negative_rules_exist():
    """System must have negative CF rules for LOW risk to be reachable."""
    negative_rules = [r for r in CF_RULES if r['cf'] < 0]
    assert len(negative_rules) > 0, "No negative CF rules found"


def test_result_structure():
    """Diagnosis result must have all required fields."""
    patient = {
        'Age': 30, 'SystolicBP': 120, 'DiastolicBP': 80,
        'BS': 7.0, 'BodyTemp': 98.6, 'HeartRate': 75
    }
    result = diagnose_patient(patient)

    required_fields = ['verdict', 'final_cf', 'nn_prob_high', 'nn_probs', 'rules_fired', 'chain']
    for field in required_fields:
        assert field in result, f"Missing field: {field}"

    assert isinstance(result['rules_fired'], list)
    assert isinstance(result['chain'], list)
    assert isinstance(result['nn_probs'], dict)
