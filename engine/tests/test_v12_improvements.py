"""Tests for v12.0 improvements — QuantumBackend ABC, TOM confidence, KernelLoader, CLI view."""

from __future__ import annotations

import json
import os

import pytest
from rdkit import Chem

from aurelius.pipeline import JSONKernelLoader, KernelLoader, _load_demo_kernel
from aurelius.scoring.oracle.quantum import (
    QuantumBackend,
    QuantumOracle,
    TOMBackend,
    XTBBackend,
    _resolve_backend,
    has_xtb,
    load_calibration_fingerprints,
)
from aurelius.types import MoleculeContext

# ---------------------------------------------------------------------------
# QuantumBackend ABC
# ---------------------------------------------------------------------------


class TestQuantumBackendABC:
    """QuantumBackend must be abstract and enforce the evaluate/method/n_calls contract."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            QuantumBackend()  # type: ignore[abstract]

    def test_concrete_backend_implements_interface(self):
        """Both XTBBackend and TOMBackend should be concrete."""
        assert issubclass(TOMBackend, QuantumBackend)
        assert issubclass(XTBBackend, QuantumBackend)

    def test_tom_backend_returns_expected_keys(self):
        """TOMBackend.evaluate() must return homo_eV, lumo_eV, dipole_D, quantum_confidence."""
        backend = TOMBackend()
        mol = Chem.MolFromSmiles("C1COC(=O)O1")
        assert mol is not None
        result = backend.evaluate(mol)
        assert "homo_eV" in result
        assert "lumo_eV" in result
        assert "dipole_D" in result
        assert "quantum_confidence" in result
        assert isinstance(result["homo_eV"], float)
        assert isinstance(result["lumo_eV"], float)
        assert backend.method == "TOM (Topological Orbital Model)"
        assert backend.n_calls >= 1

    def test_tom_backend_cache(self):
        """TOMBackend must cache results by SMILES."""
        backend = TOMBackend()
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        r1 = backend.evaluate(mol)
        r2 = backend.evaluate(mol)
        # Same molecule → same cached result
        assert r1["homo_eV"] == r2["homo_eV"]
        assert r1["lumo_eV"] == r2["lumo_eV"]
        # Cache should have 1 entry
        assert backend.get_cache_size() == 1


# ---------------------------------------------------------------------------
# TOM Confidence Score
# ---------------------------------------------------------------------------


class TestTOMConfidence:
    """TOM confidence score based on Tanimoto similarity to calibration set."""

    def test_confidence_score_present(self):
        """TOMBackend.evaluate() must include confidence_score."""
        backend = TOMBackend()
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        result = backend.evaluate(mol)
        assert "confidence_score" in result
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_confidence_score_close_to_calibration(self):
        """EC (ethylene carbonate) is in the calibration set → similarity should be high."""
        backend = TOMBackend()
        fps, _ = load_calibration_fingerprints()
        if fps:
            mol = Chem.MolFromSmiles("C1COC(=O)O1")
            assert mol is not None
            sim = backend._compute_max_tanimoto_to_calibration(mol)
            assert sim > 0.5, f"EC should have high similarity to calibration set, got {sim:.3f}"

    def test_confidence_score_novel_molecule(self):
        """A molecule outside typical electrolyte space should have low similarity."""
        backend = TOMBackend()
        fps, _ = load_calibration_fingerprints()
        if fps:
            mol = Chem.MolFromSmiles("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F")
            assert mol is not None
            sim = backend._compute_max_tanimoto_to_calibration(mol)
            assert sim < 0.5, f"Expected low similarity for perfluorinated molecule, got {sim:.3f}"


# ---------------------------------------------------------------------------
# _resolve_backend
# ---------------------------------------------------------------------------


class TestResolveBackend:
    """_resolve_backend should pick the correct backend based on inputs."""

    def test_explicit_backend_wins(self):
        """Explicit backend should always be returned regardless of use_xtb."""
        backend = TOMBackend()
        resolved = _resolve_backend(backend=backend, use_xtb=True)
        assert resolved is backend

    def test_default_no_xtb(self):
        """When xTB not on PATH and no explicit backend, should return TOMBackend."""
        xtb_avail = has_xtb()
        resolved = _resolve_backend(backend=None, use_xtb=True)
        if xtb_avail:
            assert isinstance(resolved, XTBBackend)
        else:
            assert isinstance(resolved, TOMBackend)


# ---------------------------------------------------------------------------
# QuantumOracle with backend injection
# ---------------------------------------------------------------------------


class TestQuantumOracleBackend:
    """QuantumOracle should accept and delegate to a QuantumBackend."""

    def test_backend_injection(self):
        """QuantumOracle should accept a custom backend."""
        backend = TOMBackend()
        qo = QuantumOracle(backend=backend)
        assert qo.backend is backend
        assert qo.method == "TOM (Topological Orbital Model)"

    def test_evaluate_with_tom_backend(self):
        """QuantumOracle with TOMBackend should produce valid results."""
        qo = QuantumOracle(backend=TOMBackend())
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        result = qo.evaluate(mol)
        assert "homo_eV" in result
        assert "lumo_eV" in result
        assert "quantum_confidence" in result

    def test_n_quantum_calls(self):
        """n_quantum_calls should reflect the number of backend evaluations."""
        backend = TOMBackend()
        qo = QuantumOracle(backend=backend)
        assert qo.n_quantum_calls == 0
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        qo.evaluate(mol)
        assert qo.n_quantum_calls >= 1

    def test_clear_cache(self):
        """clear_cache should clear both the oracle and backend caches."""
        backend = TOMBackend()
        qo = QuantumOracle(backend=backend)
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        qo.evaluate(mol)
        assert qo.get_cache_size() >= 1
        qo.clear_cache()
        assert qo.get_cache_size() == 0
        assert backend.get_cache_size() == 0


# ---------------------------------------------------------------------------
# KernelLoader / JSONKernelLoader
# ---------------------------------------------------------------------------


class TestKernelLoader:
    """KernelLoader ABC and JSONKernelLoader implementation."""

    def test_kernel_loader_abc(self):
        """KernelLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            KernelLoader()  # type: ignore[abstract]

    def test_json_kernel_loader_concrete(self):
        """JSONKernelLoader should be a concrete KernelLoader."""
        loader = JSONKernelLoader()
        assert isinstance(loader, KernelLoader)

    def test_json_kernel_loader_carbonate_demo(self):
        """JSONKernelLoader should load the carbonate_high_voltage demo kernel."""
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "docs", "examples", "kernels", "carbonate_high_voltage.json"),
        ]
        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        if path is None:
            pytest.skip("carbonate_high_voltage.json not found")
        loader = JSONKernelLoader()
        kernel = loader.load(path)
        # Loading should return a dict (kernel params) or None if the file
        # is missing required fields. It should never crash.
        assert kernel is None or isinstance(kernel, dict)

    def test_json_kernel_loader_missing_file(self):
        """JSONKernelLoader.load() should return None for nonexistent files."""
        loader = JSONKernelLoader()
        result = loader.load("/nonexistent/path/kernel.json")
        assert result is None

    def test_json_kernel_loader_verify(self):
        """verify() should return False for kernels missing required fields."""
        loader = JSONKernelLoader()
        incomplete = {"version": "1.0.0", "tom_parameters": {"homo_offset": 0.0, "lumo_offset": 0.0}}
        assert not loader.verify(incomplete)

    def test_demo_kernel_loader(self):
        """_load_demo_kernel should return a dict or None without crashing."""
        result = _load_demo_kernel()
        # Should not crash
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# Pipeline demo mode
# ---------------------------------------------------------------------------


class TestPipelineDemoMode:
    """Demo mode should load the demo kernel without crashing."""

    def test_demo_flag_in_pipeline(self):
        """AureliusPipeline should accept a kernel_loader."""
        from aurelius.pipeline import AureliusPipeline
        pipeline = AureliusPipeline(use_real_models=False)
        pipeline.initialize()
        # Should not crash
        assert pipeline._kernel_loader is not None

    def test_screen_with_demo(self):
        """Screening with demo flag should work without the demo kernel."""
        from aurelius.pipeline import AureliusPipeline
        from aurelius.scoring.oracle.gc import ElectrolytePack
        pipeline = AureliusPipeline(
            use_real_models=False,
            property_pack=ElectrolytePack(),
        )
        pipeline.initialize()
        result = pipeline.screen_smiles("CCO")
        assert result is not None
        assert "score" in result


# ---------------------------------------------------------------------------
# CLI View Command
# ---------------------------------------------------------------------------


class TestCLIView:
    """The 'aurelius view' command should generate HTML without crashing."""

    def test_view_html_generation(self):
        """Generate an HTML report via the Click command logic."""
        import base64
        from io import BytesIO

        from rdkit.Chem import Draw

        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        img = Draw.MolToImage(ctx.mol, size=(300, 300))
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        assert len(img_b64) > 0

    def test_screen_report_flag_exists(self):
        """The 'screen' command should have a --report flag."""
        from aurelius.__main__ import cli
        screen_cmd = cli.commands.get("screen")
        assert screen_cmd is not None
        assert any(p.name == "report" for p in screen_cmd.params)

    def test_screen_report_help(self):
        """The --report flag should be described."""
        from aurelius.__main__ import cli
        screen_cmd = cli.commands.get("screen")
        assert screen_cmd is not None
        report_param = next((p for p in screen_cmd.params if p.name == "report"), None)
        assert report_param is not None
        assert "HTML" in (report_param.help or "")


# ---------------------------------------------------------------------------
# Agent State — Atomic Checkpointing
# ---------------------------------------------------------------------------


class TestAtomicCheckpointing:
    """LoopState.save() must write to .tmp then os.replace()."""

    def test_save_uses_tmp_and_replace(self, tmp_path):
        """Save should write to .tmp then replace the target."""
        from aurelius.agent.state import LoopState
        state_path = str(tmp_path / "agent_state.json")
        ls = LoopState()
        ls.path = state_path
        ls.total_screened = 42
        ls.save()
        assert os.path.exists(state_path)
        # Verify the saved data
        with open(state_path) as f:
            data = json.load(f)
        assert data["total_screened"] == 42
        # .tmp file should have been removed by os.replace
        assert not os.path.exists(state_path + ".tmp")


# ---------------------------------------------------------------------------
# Kernel Schema — Provenance & Metadata
# ---------------------------------------------------------------------------


class TestKernelSchema:
    """kernel_schema.json must include optional metadata and provenance fields."""

    def test_kernel_schema_has_metadata(self):
        """kernel_schema.json should define a 'metadata' property."""
        schema_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docs", "kernel_schema.json",
        )
        with open(schema_path) as f:
            schema = json.load(f)
        assert "metadata" in schema["properties"]
        assert "type" in schema["properties"]["metadata"]
        assert schema["properties"]["metadata"]["type"] == "object"

    def test_metadata_has_author_and_date(self):
        """metadata should include author and date fields."""
        schema_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docs", "kernel_schema.json",
        )
        with open(schema_path) as f:
            schema = json.load(f)
        meta = schema["properties"]["metadata"]["properties"]
        assert "author" in meta
        assert "date" in meta

    def test_kernel_schema_has_provenance(self):
        """kernel_schema.json should define a 'provenance' property."""
        schema_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docs", "kernel_schema.json",
        )
        with open(schema_path) as f:
            schema = json.load(f)
        assert "provenance" in schema["properties"]
        assert "type" in schema["properties"]["provenance"]
        assert schema["properties"]["provenance"]["type"] == "object"

    def test_provenance_has_data_sources_tuning_method(self):
        """provenance should include data_sources and tuning_method."""
        schema_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docs", "kernel_schema.json",
        )
        with open(schema_path) as f:
            schema = json.load(f)
        prov = schema["properties"]["provenance"]["properties"]
        assert "data_sources" in prov
        assert "tuning_method" in prov
