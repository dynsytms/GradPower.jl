"""Focused tests for Phase 0 reference metadata and state manifests."""

import json
from types import SimpleNamespace

import numpy as np

from reference_common import (
    ReferenceCase,
    _fault_bus,
    _integration_config,
    _json_bytes,
    _metadata,
    _power_flow_data,
    build_parameter_manifest,
    build_state_manifest,
)
from uqgrid import IntegrationConfig


def _fake_system():
    device = SimpleNamespace(
        dif_dim=2,
        alg_dim=1,
        dif_ptr=0,
        alg_ptr=0,
        par_dim=2,
        par_ptr=0,
        state_list=["x1", "x2", "y1"],
        model_type="generator",
        id_tag="1",
        bus=0,
    )
    return SimpleNamespace(
        num_dof_dif=2,
        num_dof_alg=1,
        nbuses=1,
        devices=[device],
        buses=[SimpleNamespace(id=7, i=7)],
        ext2int={7: 0},
        power_injection=False,
    )


def test_manifest_covers_every_history_row_once():
    manifest = build_state_manifest(_fake_system(), history_rows=5)
    assert [entry["index"] for entry in manifest] == list(range(5))
    assert [entry["partition"] for entry in manifest] == [
        "differential",
        "differential",
        "algebraic",
        "network",
        "network",
    ]
    assert [entry["name"] for entry in manifest] == [
        "x1",
        "x2",
        "y1",
        "voltage_real",
        "voltage_imaginary",
    ]
    assert manifest[0]["canonical_model"] == "simplenamespace"
    assert manifest[-1]["canonical_model"] == "bus"
    assert {entry["bus_external"] for entry in manifest} == {7}


def test_metadata_records_reproducibility_contract():
    case = ReferenceCase(
        name="test",
        raw="examples/2bus.raw",
        dyr="examples/2bus.dyr",
        output="unused.npz",
        generator_script="scripts/gen_ref_2bus_genrou.py",
    )
    config = IntegrationConfig(
        method="beuler", enforce_dynamic_limits=False, enforce_q_limits=False
    )
    metadata = _metadata(case, config, build_state_manifest(_fake_system(), 5))
    assert metadata["integration"]["method"] == "beuler"
    assert metadata["integration"]["enforce_dynamic_limits"] is False
    assert metadata["integration"]["enforce_q_limits"] is False
    assert metadata["uqgrid_git"].keys() >= {"commit", "tree", "dirty"}
    assert all(item["sha256"] for item in metadata["source_files"].values())
    encoded = _json_bytes(metadata)
    assert encoded.dtype == np.uint8
    assert json.loads(encoded.tobytes().decode("utf-8"))["case"] == "test"


def test_parameter_manifest_records_device_slices():
    manifest = build_parameter_manifest(_fake_system())
    assert manifest == [{
        "start_index": 0,
        "count": 2,
        "device_class": "SimpleNamespace",
        "canonical_model": "simplenamespace",
        "device_id": "1",
        "bus_internal": 0,
        "bus_external": 7,
    }]


def test_phase_zero_config_explicitly_disables_limits():
    config = _integration_config(
        ReferenceCase("test", "raw", "dyr", "out", "generator")
    )
    assert config.enforce_dynamic_limits is False
    assert config.enforce_q_limits is False
    assert config.method == "beuler"


def test_external_fault_bus_resolves_internal_index():
    case = ReferenceCase(
        "test",
        "raw",
        "dyr",
        "out",
        "generator",
        fault_bus_external=1001,
    )
    assert _fault_bus(case, SimpleNamespace(ext2int={1001: 17})) == (
        17,
        1001,
        1001,
    )


def test_internal_fault_bus_preserves_legacy_value_and_resolves_external_id():
    case = ReferenceCase("test", "raw", "dyr", "out", "generator", fault_bus=6)
    assert _fault_bus(case, SimpleNamespace(ext2int={7: 6})) == (6, 7, 6)


def test_power_flow_data_preserves_legacy_activs_layout():
    generators = [
        SimpleNamespace(bus=1, id_tag="'ABCD5'", dif_ptr=3, alg_ptr=7),
    ]
    psys = SimpleNamespace(ext2int={10: 0, 20: 1}, nbuses=2, gendyn=generators)
    pf = SimpleNamespace(v_magnitudes=[1.0, 0.9], v_angles=[0.0, -0.1])
    z0 = np.array([1.0, 2.0])

    data = _power_flow_data(psys, pf, z0)

    np.testing.assert_array_equal(data["bus_ids"], [10, 20])
    np.testing.assert_array_equal(data["gen_bus"], [20])
    np.testing.assert_array_equal(data["gen_id"], [[65, 66, 67, 68]])
    np.testing.assert_array_equal(data["gen_dif_ptr"], [3])
    np.testing.assert_array_equal(data["gen_alg_ptr"], [7])
    assert data["z0"] is z0
    assert data["n_dyn_gen"] == 1
