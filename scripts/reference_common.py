"""Shared infrastructure for small uqgrid reference generators."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
UQGRID_REPO = REPO / "uqgrid"
if str(UQGRID_REPO) not in sys.path:
    sys.path.insert(0, str(UQGRID_REPO))

import uqgrid  # noqa: E402
from uqgrid import IntegrationConfig, add_dyr, integrate_system, load_psse  # noqa: E402
from uqgrid.simulation.dynamics import initialize_system  # noqa: E402
from uqgrid.simulation.pflow import runpf  # noqa: E402


_CANONICAL_MODELS = {
    "GenGENROU": "genrou",
    "GenGENSAL": "genrou",
    "GovIEESGO": "ieesgo",
    "GovTGOV1": "tgov1",
    "ExcSEXS": "sexs",
    "ExcESDC1A": "esdc1a",
    "PssIEEEST": "ieeest",
    "StaticGenerator": "staticgenerator",
}


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    raw: str
    dyr: str
    output: str
    generator_script: str
    fault_bus: int = 0
    fault_bus_external: int | None = None
    rfault: float = 0.02
    ton: float = 0.2
    toff: float = 0.3
    dt: float = 1.0 / 120.0
    tend: float = 5.0
    zipload_alpha: float = 0.5
    include_power_flow_data: bool = False
    dyr_limit_initialization_policy: str = "adjust"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(UQGRID_REPO), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_state_manifest(psys: Any, history_rows: int) -> list[dict[str, Any]]:
    """Describe every row in uqgrid's [differential, algebraic, network] vector."""
    expected_rows = int(psys.num_dof_dif) + int(psys.num_dof_alg) + 2 * int(psys.nbuses)
    if history_rows != expected_rows:
        raise ValueError(
            f"history has {history_rows} rows, but system dimensions require {expected_rows}"
        )
    manifest: list[dict[str, Any] | None] = [None] * history_rows
    dif_size = int(psys.num_dof_dif)
    alg_size = int(psys.num_dof_alg)
    external_bus = {
        int(internal): int(external)
        for external, internal in getattr(psys, "ext2int", {}).items()
    }

    for device in psys.devices:
        names = list(getattr(device, "state_list", ()))
        expected = int(device.dif_dim) + int(device.alg_dim)
        if expected and len(names) != expected:
            raise ValueError(
                f"{type(device).__name__} exposes {len(names)} state names for "
                f"{expected} states"
            )
        common = {
            "device_class": type(device).__name__,
            "device_type": str(device.model_type),
            "canonical_model": _CANONICAL_MODELS.get(
                type(device).__name__, type(device).__name__.lower()
            ),
            "device_id": str(device.id_tag).strip(),
            "bus_internal": int(device.bus),
            "bus_external": external_bus.get(int(device.bus), int(psys.buses[device.bus].i)),
        }
        for offset in range(int(device.dif_dim)):
            index = int(device.dif_ptr) + offset
            if manifest[index] is not None:
                raise ValueError(f"multiple states map to history row {index}")
            manifest[index] = {
                "index": index,
                "partition": "differential",
                "name": names[offset],
                **common,
            }
        for offset in range(int(device.alg_dim)):
            index = dif_size + int(device.alg_ptr) + offset
            if manifest[index] is not None:
                raise ValueError(f"multiple states map to history row {index}")
            manifest[index] = {
                "index": index,
                "partition": "algebraic",
                "name": names[int(device.dif_dim) + offset],
                **common,
            }

    network_start = dif_size + alg_size
    coordinate_names = ("voltage_magnitude", "voltage_angle") if psys.power_injection else (
        "voltage_real",
        "voltage_imaginary",
    )
    for bus_internal, bus in enumerate(psys.buses):
        for offset, name in enumerate(coordinate_names):
            index = network_start + 2 * bus_internal + offset
            if manifest[index] is not None:
                raise ValueError(f"multiple states map to history row {index}")
            manifest[index] = {
                "index": index,
                "partition": "network",
                "name": name,
                "device_class": None,
                "device_type": "bus",
                "canonical_model": "bus",
                "device_id": str(external_bus.get(bus_internal, int(bus.i))),
                "bus_internal": bus_internal,
                "bus_external": external_bus.get(bus_internal, int(bus.i)),
            }

    missing = [index for index, entry in enumerate(manifest) if entry is None]
    if missing:
        raise ValueError(f"state manifest does not cover history rows {missing}")
    return manifest  # type: ignore[return-value]


def _json_bytes(value: Any) -> np.ndarray:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def build_parameter_manifest(psys: Any) -> list[dict[str, Any]]:
    """Describe each device's slice in the initialized parameter vector."""
    external_bus = {
        int(internal): int(external)
        for external, internal in getattr(psys, "ext2int", {}).items()
    }
    return [
        {
            "start_index": int(device.par_ptr),
            "count": int(device.par_dim),
            "device_class": type(device).__name__,
            "canonical_model": _CANONICAL_MODELS.get(
                type(device).__name__, type(device).__name__.lower()
            ),
            "device_id": str(device.id_tag).strip(),
            "bus_internal": int(device.bus),
            "bus_external": external_bus.get(int(device.bus), int(psys.buses[device.bus].i)),
        }
        for device in psys.devices
        if int(device.par_dim) > 0
    ]


def _metadata(
    case: ReferenceCase,
    config: IntegrationConfig,
    manifest: list[dict[str, Any]],
    parameter_manifest: list[dict[str, Any]] | None = None,
    dynamic_limit_diagnostics: Any = None,
    fault_bus_internal: int | None = None,
    fault_bus_external: int | None = None,
) -> dict[str, Any]:
    source_paths = {
        "raw": REPO / case.raw,
        "dyr": REPO / case.dyr,
        "generator": REPO / case.generator_script,
        "common": Path(__file__).resolve(),
    }
    return {
        "schema_version": 1,
        "state_index_base": 0,
        "case": case.name,
        "source_files": {
            role: {
                "path": str(path.relative_to(REPO)),
                "sha256": _sha256(path),
            }
            for role, path in source_paths.items()
        },
        "uqgrid_git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "tree": _git_value("rev-parse", "HEAD^{tree}"),
            "dirty": bool(_git_value("status", "--porcelain=v1")),
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                **{
                    name: _package_version(name)
                    for name in ("numpy", "scipy", "numba", "pydantic")
                },
                "uqgrid": _package_version("uqgrid") or uqgrid.__version__,
            },
            "threading_env": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
        },
        "integration": config.model_dump(mode="json"),
        "dyr": {
            "limit_initialization_policy": case.dyr_limit_initialization_policy,
        },
        "fault": {
            "bus_internal": (
                case.fault_bus if fault_bus_internal is None else fault_bus_internal
            ),
            "bus_external": (
                case.fault_bus_external
                if fault_bus_external is None
                else fault_bus_external
            ),
        },
        "limits": {
            "dynamic": "disabled",
            "reactive_power": "disabled during initial power flow",
        },
        "event_sampling": {
            "grid": "nominal dt grid with exact ton/toff inserted",
            "event_sample": "post-event algebraic projection at the event time",
            "step_to_event": "uses the pre-event topology",
            "initial_sample": "initialized pre-fault state at t=0",
        },
        "state_manifest": manifest,
        "parameter_manifest": parameter_manifest or [],
        "dynamic_limit_diagnostics": dynamic_limit_diagnostics,
    }


def _fault_bus(case: ReferenceCase, psys: Any) -> tuple[int, int, int]:
    """Return internal, external, and legacy NPZ fault identities."""
    if case.fault_bus_external is None:
        int2ext = {internal: external for external, internal in psys.ext2int.items()}
        return case.fault_bus, int(int2ext[case.fault_bus]), case.fault_bus
    return (
        int(psys.ext2int[case.fault_bus_external]),
        case.fault_bus_external,
        case.fault_bus_external,
    )


def _power_flow_data(psys: Any, pf: Any, z0: np.ndarray) -> dict[str, Any]:
    """Build the legacy ACTIVS power-flow and dynamic-generator payload."""
    int2ext = {internal: external for external, internal in psys.ext2int.items()}
    raw_ids = [str(gen.id_tag).strip().replace("'", "") for gen in psys.gendyn]
    gen_id = np.zeros((len(raw_ids), 4), dtype=np.uint8)
    for index, identifier in enumerate(raw_ids):
        encoded = identifier.encode("ascii")[:4]
        gen_id[index, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)

    return {
        "bus_ids": np.array(
            [int2ext[index] for index in range(psys.nbuses)], dtype=np.int64
        ),
        "pf_vmag": np.array(pf.v_magnitudes),
        "pf_vang": np.array(pf.v_angles),
        "z0": z0,
        "gen_bus": np.array(
            [int(int2ext[gen.bus]) for gen in psys.gendyn], dtype=np.int64
        ),
        "gen_id": gen_id,
        "gen_dif_ptr": np.array(
            [gen.dif_ptr for gen in psys.gendyn], dtype=np.int64
        ),
        "gen_alg_ptr": np.array(
            [gen.alg_ptr for gen in psys.gendyn], dtype=np.int64
        ),
        "n_dyn_gen": len(psys.gendyn),
    }


def _integration_config(case: ReferenceCase) -> IntegrationConfig:
    """Create the unlimited Phase 0 integration configuration."""
    return IntegrationConfig(
        tend=case.tend,
        dt=case.dt,
        ton=case.ton,
        toff=case.toff,
        method="beuler",
        power_injection=False,
        verbose=False,
        comp_sens=False,
        petsc=False,
        enforce_dynamic_limits=False,
        enforce_q_limits=False,
    )


def generate_reference(case: ReferenceCase) -> None:
    """Run one declared case and write its backward-compatible enriched NPZ."""
    raw = REPO / case.raw
    dyr = REPO / case.dyr
    output = REPO / case.output

    psys = load_psse(raw_filename=str(raw))
    add_dyr(
        psys,
        str(dyr),
        limit_initialization_policy=case.dyr_limit_initialization_policy,
    )
    fault_bus_internal, fault_bus_external, fault_bus_output = _fault_bus(case, psys)
    if not case.include_power_flow_data:
        psys.add_busfault(fault_bus_internal, case.rfault)
    psys.createYbusComplex()
    psys.set_load_parameters(np.full(psys.nloads, case.zipload_alpha))

    legacy_data: dict[str, Any] = {}
    if case.include_power_flow_data:
        pf = runpf(psys, verbose=False, enforce_q_limits=False)
        legacy_z0, _ = initialize_system(psys, pf)
        legacy_data = _power_flow_data(psys, pf, legacy_z0)
        psys.add_busfault(fault_bus_internal, case.rfault)

    config = _integration_config(case)
    results = integrate_system(psys, config)
    history = results["history"]
    z0 = np.array(history[:, 0], copy=True)
    theta = np.zeros(psys.num_pars)
    for device in psys.devices:
        device.initialize_theta(theta)

    manifest = build_state_manifest(psys, history.shape[0])
    parameter_manifest = build_parameter_manifest(psys)
    metadata = _metadata(
        case,
        config,
        manifest,
        parameter_manifest,
        results.get("dynamic_limit_diagnostics"),
        fault_bus_internal,
        fault_bus_external,
    )
    output_z0 = legacy_data.pop("z0", z0)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        tvec=results["tvec"],
        history=history,
        speed_idx=np.array(psys.genspeed_idx_set()),
        n_diff=psys.num_dof_dif,
        n_alg=psys.num_dof_alg,
        n_bus=psys.nbuses,
        fault_bus=fault_bus_output,
        fault_bus_internal=fault_bus_internal,
        fault_bus_external=fault_bus_external,
        rfault=case.rfault,
        ton=case.ton,
        toff=case.toff,
        dt=case.dt,
        tend=case.tend,
        zipload_alpha=case.zipload_alpha,
        z0=output_z0,
        theta=theta,
        state_manifest_json=_json_bytes(manifest),
        metadata_json=_json_bytes(metadata),
        **legacy_data,
    )
    print(
        f"Saved: {output} history.shape={history.shape} "
        f"n_diff={psys.num_dof_dif} n_alg={psys.num_dof_alg} n_bus={psys.nbuses}"
    )
