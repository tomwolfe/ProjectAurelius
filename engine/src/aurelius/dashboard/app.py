"""Aurelius Discovery Dashboard — Streamlit application.

Tabs:
1. Discovery Trajectory: score vs generation, scaffold novelty vs time
2. Chemical Space: UMAP of discovered molecules
3. Pareto Front: interactive 3D plot
4. Molecule Viewer: RDKit 2D with property annotations
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st
import numpy as np

logger = logging.getLogger(__name__)


def _load_discoveries() -> list[dict]:
    """Load discovery data from JSON files."""
    paths = [
        Path("agent_state.json"),
        Path("discovery_results.json"),
        Path("discovery_results_final.json"),
        Path("run_summary.json"),
    ]
    for path in paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if "discoveries" in data:
                return data["discoveries"]
            if "all_results" in data:
                return data["all_results"]
    return []


def _load_pareto() -> list[dict]:
    """Load Pareto front data."""
    path = Path("pareto_front.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _load_benchmark() -> list[dict]:
    """Load external property benchmark data."""
    path = Path("external_property_benchmark.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _load_calibration() -> list[dict]:
    """Load orbital calibration data."""
    path = Path("orbital_calibration.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _smiles_to_rdkit(smiles: str) -> Any | None:
    """Parse SMILES into RDKit Mol object."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _compute_umap(data: list[list[float]], n_components: int = 2) -> list[list[float]]:
    """Compute UMAP embedding from feature vectors."""
    if len(data) < 2:
        return data
    arr = np.array(data)
    if arr.shape[1] < 2:
        return [[float(x)] for x in data]
    # Simple PCA as fallback for UMAP
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, arr.shape[1]))
    return pca.fit_transform(arr).tolist()


def _compute_pca(data: list[list[float]], n_components: int = 2) -> list[list[float]]:
    """Compute PCA embedding as fallback for UMAP."""
    if len(data) < 2:
        return data
    arr = np.array(data)
    if arr.shape[1] < 2:
        return [[float(x)] for x in data]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, arr.shape[1]))
    return pca.fit_transform(arr).tolist()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Aurelius Discovery Dashboard."""
    st.set_page_config(page_title="Aurelius Discovery Dashboard", layout="wide")
    st.title("🔬 Project Aurelius — Discovery Dashboard")

    # Sidebar
    st.sidebar.header("Navigation")
    tabs = [
        ("Discovery Trajectory", "scores"),
        ("Chemical Space", "umap"),
        ("Pareto Front", "pareto"),
        ("Molecule Viewer", "viewer"),
    ]
    selected_tab = st.sidebar.radio("Select tab", [t[0] for t in tabs])

    # Load data
    discoveries = _load_discoveries()
    pareto = _load_pareto()
    benchmark = _load_benchmark()
    calibration = _load_calibration()

    # Render selected tab
    if selected_tab == "Discovery Trajectory":
        _render_discovery_trajectory(discoveries)
    elif selected_tab == "Chemical Space":
        _render_chemical_space(discoveries, benchmark)
    elif selected_tab == "Pareto Front":
        _render_pareto_front(pareto)
    elif selected_tab == "Molecule Viewer":
        _render_molecule_viewer(discoveries, calibration)


def _render_discovery_trajectory(discoveries: list[dict]) -> None:
    """Render discovery trajectory: score vs generation, scaffold novelty vs time."""
    st.header("Discovery Trajectory")

    if not discoveries:
        st.info("No discovery data loaded. Run the agent first to generate discoveries.")
        return

    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame(discoveries)
    if "total_score" in df.columns:
        fig = px.line(df, x=range(len(df)), y="total_score",
                       title="Discovery Score Trajectory",
                       labels={"x": "Generation", "y": "Score"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No score data available for trajectory visualization.")


def _render_chemical_space(discoveries: list[dict], benchmark: list[dict]) -> None:
    """Render chemical space: UMAP of discovered molecules."""
    st.header("Chemical Space Exploration")

    if not discoveries:
        st.info("No discovery data loaded.")
        return

    try:
        import umap
        import numpy as np

        # Extract fingerprints or features
        features = []
        for d in discoveries[:100]:  # Limit for performance
            smiles = d.get("smiles", "")
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVector(mol, radius=2, nBits=2048)
                    features.append([fp.GetNumOnBits()])  # Simplified for demo
            except Exception:
                features.append([0.0])

        if len(features) >= 2:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(features)

            import plotly.express as px
            df = pd.DataFrame(embedding, columns=["x", "y"])
            df["score"] = [d.get("total_score", 0.0) for d in discoveries[:100]]
            df["smiles"] = [d.get("smiles", "") for d in discoveries[:100]]

            fig = px.scatter(df, x="x", y="y", color="score",
                             color_continuous_scale="Viridis",
                             hover_data=["smiles"],
                             title="Chemical Space (UMAP embedding)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for UMAP visualization.")
    except ImportError:
        st.warning("umap package required. Install with: pip install umap-learn")


def _render_pareto_front(pareto: list[dict]) -> None:
    """Render Pareto front: interactive 3D plot."""
    st.header("Pareto Front Visualization")

    if not pareto:
        st.info("No Pareto front data loaded.")
        return

    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(pareto)
        if "lumo_eV" in df.columns and "dielectric_proxy" in df.columns and "viscosity_proxy" in df.columns:
            fig = px.scatter_3d(df, x="lumo_eV", y="dielectric_proxy", z=-df["viscosity_proxy"].astype(float),
                                color="total_score", color_continuous_scale="Viridis",
                                hover_data=["smiles"],
                                title="Pareto Front (3D: LUMO, Dielectric, -Viscosity)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data for 3D Pareto visualization.")
    except ImportError:
        st.warning("plotly package required for visualization.")


def _render_molecule_viewer(discoveries: list[dict], calibration: list[dict]) -> None:
    """Render molecule viewer with RDKit 2D and property annotations."""
    st.header("Molecule Viewer")

    if not discoveries:
        st.info("No discovery data loaded.")
        return

    # Select molecule to view
    smi_options = [d.get("smiles", "") for d in discoveries if d.get("smiles")]
    if not smi_options:
        st.warning("No SMILES found in discovery data.")
        return

    selected_smiles = st.selectbox("Select molecule", smi_options)

    if selected_smiles:
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(selected_smiles)
            if mol is None:
                st.error(f"Invalid SMILES: {selected_smiles}")
                return

            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)

            # Get properties
            props = {d.get("smiles", ""): d for d in discoveries if d.get("smiles") == selected_smiles}
            if props:
                props = props[selected_smiles]

            # Draw molecule
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, use_container_width=True)

            # Display properties
            st.subheader("Properties")
            for key, val in props.items():
                if key != "smiles":
                    st.write(f"**{key}**: {val}")

        except Exception as e:
            st.error(f"Error rendering molecule: {e}")


if __name__ == "__main__":
    main()
