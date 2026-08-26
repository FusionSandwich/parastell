from parastell.production_deck import (
    REQUIRED_GATES,
    evaluate_production_deck_gate,
)


def test_production_deck_gate_fails_closed():
    gates = {name: "PASS" for name in REQUIRED_GATES}
    gates["source_mesh_convergence"] = "SOURCE_MESH_CONVERGENCE_NOT_REACHED"
    result = evaluate_production_deck_gate(gates)
    assert result["status"] == "BLOCKED_GATES_FAILED"
    assert not result["ready"]
    assert result["nonpassing_gates"] == ["source_mesh_convergence"]
    assert not result["submission_authorized"]


def test_production_deck_requires_every_gate_and_separate_authorization():
    result = evaluate_production_deck_gate(
        {name: True for name in REQUIRED_GATES}
    )
    assert result["status"] == "PRODUCTION_DECK_READY"
    assert result["ready"]
    assert not result["submission_authorized"]


def test_missing_gate_is_not_treated_as_pass():
    result = evaluate_production_deck_gate({})
    assert result["missing_gates"] == list(REQUIRED_GATES)
    assert not result["ready"]


def test_integer_one_is_not_accepted_as_boolean_pass():
    gates = {name: True for name in REQUIRED_GATES}
    gates["geometry_full_assembly"] = 1
    result = evaluate_production_deck_gate(gates)
    assert result["nonpassing_gates"] == ["geometry_full_assembly"]
    assert not result["ready"]
