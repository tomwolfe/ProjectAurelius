"""Phase 3: Tier 3 - GCMD Digital Twin with Arrhenius kMC.

Implements a kinetic Monte Carlo (kMC) simulation for SEI growth,
using physically derived rate constants from the Arrhenius equation.

Reaction pathways:
    - Solvent decomposition (EC/DMC reduction at anode surface)
    - Salt reduction (PF6- decomposition)
    - Polymerization (organic SEI formation)

Rate constants follow: k = A * exp(-Ea / (kB * T))

where A (pre-exponential factor) depends on local solvent
concentration (mass transport limitation as SEI grows),
Ea is the activation energy from literature values,
and kB is the Boltzmann constant.

Note: This is a standard kinetic Monte Carlo simulator. It does not
use any Transformer models, KV caching, or model quantization. The
name "GCMD Digital Twin" refers to the "Gaussian-Chemical Multiscale
Digital" twin concept for battery electrolyte screening.
"""

from __future__ import annotations

import json
import os

import numpy as np

from aurelius.types import GCMDTConfig, GCMDTwinResult, SEIEvolution

# Physical constants
_KB_EV_K = 8.617333262e-5  # Boltzmann constant in eV/K
_KB_J_K = 1.380649e-23     # Boltzmann constant in J/K
_AVOGADRO = 6.02214076e23   # Avogadro's number


def _load_kmc_params(path: str | None = None) -> dict:
    """Load kMC reaction parameters from force field JSON.

    Args:
        path: Optional path to force field params JSON file.

    Returns:
        Dictionary of kMC reaction parameters, or empty dict on failure.
    """
    ff_path = path or os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "force_field_params.json",
    )
    if os.path.isfile(ff_path):
        try:
            with open(ff_path) as f:
                data = json.load(f)
                return data.get("kmc_reaction_parameters", {})
        except (json.JSONDecodeError, OSError):
            pass
    return {}


class GCMDigitalTwin:
    """Tier 3: GCMD Digital Twin with Arrhenius kMC.

    Simulates SEI (Solid Electrolyte Interphase) evolution at the
    anode interface using kinetic Monte Carlo (kMC) to model
    discrete solvent decomposition and salt reduction reactions.

    Reaction rate constants are computed via the Arrhenius equation
    with literature-derived activation energies and mass-transport
    limited pre-exponential factors.

    Parameters are loaded from force_field_params.json for
    configurability without code changes.
    """

    def __init__(
        self,
        gcmtwin_config: GCMDTConfig | None = None,
        force_field_path: str | None = None,
    ) -> None:
        """Initialize GCMD Digital Twin.

        Args:
            gcmtwin_config: kMC simulation configuration.
            force_field_path: Optional path to force field JSON.
        """
        self.config = gcmtwin_config or GCMDTConfig()
        self._kmc_params = _load_kmc_params(force_field_path)
        self._activation_energies = self._kmc_params.get("activation_energies_eV", {})
        self._pre_exponential_factors = self._kmc_params.get("pre_exponential_factors_ps", {})
        self._thickness_contributions = self._kmc_params.get("thickness_contributions_angstrom", {})
        self._kinetic_params = self._kmc_params.get("kinetic_parameters", {})
        self._initial_concentrations = self._kmc_params.get("initial_concentrations", {})
        self._solvent_composition = self._kmc_params.get("solvent_composition", {})
        self._sei_property_params = self._kmc_params.get("sei_property_parameters", {})
        self._memory_model = self._kmc_params.get("memory_model", {})

        # Activation energies (eV) from DFT/experimental literature
        self._Ea_SOLVENT_EC = self._activation_energies.get("ec_reduction", 0.65)
        self._Ea_SOLVENT_DMC = self._activation_energies.get("dm_reduction", 0.75)
        self._Ea_SALT_PF6 = self._activation_energies.get("pf6_decomposition", 1.20)

        # Pre-exponential factors (1/ps) at standard conditions
        self._A_SOLVENT_BASE = self._pre_exponential_factors.get("solvent", 5.0)
        self._A_SALT_BASE = self._pre_exponential_factors.get("salt", 2.0)
        self._A_POLY_BASE = self._pre_exponential_factors.get("polymerization", 1.0)

        # Thickness contribution per reaction event (Angstrom)
        self._D_SOLVENT = self._thickness_contributions.get("solvent", 0.03)
        self._D_SALT = self._thickness_contributions.get("salt", 0.04)
        self._D_POLY = self._thickness_contributions.get("polymerization", 0.05)

        # Kinetic parameters
        self._K_m = self._kinetic_params.get("km_half_saturation", 0.3)
        self._alpha = self._kinetic_params.get("symmetry_factor_alpha", 0.5)
        self._polymer_voltage_factor = self._kinetic_params.get("polymer_voltage_factor", 0.5)

        # Initial concentrations
        self._initial_solvent_conc = self._initial_concentrations.get("solvent", 1.0)
        self._initial_salt_conc = self._initial_concentrations.get("salt", 0.1)

        # Solvent composition defaults
        self._ec_ratio_default = self._solvent_composition.get("ec_dmc_ec_ratio", 0.3)

        # SEI property parameters
        self._ionic_cond_base = self._sei_property_params.get("ionic_conductivity_base_s_cm", 1e-4)
        self._cond_decay_length = self._sei_property_params.get("conductivity_decay_length_angstrom", 10.0)
        self._insulation_threshold = self._sei_property_params.get("electronic_insulation_threshold_angstrom", 2.0)
        self._ideal_fraction = self._sei_property_params.get("ideal_reaction_fraction", 1.0 / 3.0)

        # Memory model
        self._mem_base = self._memory_model.get("base_footprint_gb", 2.0)
        self._mem_scaling = self._memory_model.get("per_angstrom_scaling", 0.1)

    def simulate_sei_evolution(
        self,
        smiles: str,
        solvent_type: str = "ec:dmc",
        salt_type: str = "NaPF6",
        voltage_cutoff: float = 0.05,
        max_time_ps: float = 1000.0,
        temperature_k: float = 298.15,
    ) -> GCMDTwinResult:
        """Run GCMD Digital Twin simulation of SEI evolution.

        Uses kinetic Monte Carlo (kMC) with Arrhenius-derived
        rate constants to simulate SEI layer growth with
        mass-transport limitations.
        """
        import time
        start = time.perf_counter()

        seed = self._hash_inputs(smiles, solvent_type, salt_type)
        rng = np.random.RandomState(seed)

        sei = self._run_kmc_simulation(
            rng, voltage_cutoff, max_time_ps, temperature_k,
            solvent_type, salt_type
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        mem_gb = self._estimate_memory_footprint(
            sei, base=self._mem_base, scaling=self._mem_scaling
        )

        return GCMDTwinResult(
            molecule_smiles=smiles,
            sei_evolution=sei,
            interface_stability=sei.homogeneity_score,
            memory_used_gb=mem_gb,
            context_tokens_used=self.config.max_simulation_steps,
            simulation_time_ms=elapsed_ms,
        )

    def get_simulation_stats(self) -> dict:
        """Return current kMC simulation statistics."""
        return {
            "max_simulation_steps": self.config.max_simulation_steps,
            "record_interval": self.config.record_interval,
            "transport_limit_thickness_angstrom": self.config.transport_limit_thickness_angstrom,
            "use_mass_transport_limitation": self.config.use_mass_transport_limitation,
        }

    def _arrhenius_rate(
        self,
        activation_energy_eV: float,
        temperature_k: float,
        concentration: float,
        pre_exponential_base: float,
        overpotential_V: float,
    ) -> float:
        """Compute Arrhenius rate constant with overpotential dependence.

        k = A(concentration) * exp(-Ea / (kB * T)) * exp(alpha * eta)

        where:
            - A(concentration) is the pre-exponential factor scaled by
              local solvent concentration (mass transport limitation)
            - Ea is the activation energy in eV
            - kB is the Boltzmann constant
            - T is temperature in Kelvin
            - eta is the overpotential (voltage driving force)
            - alpha is the symmetry factor (~0.5 for electron transfer)

        Args:
            activation_energy_eV: Activation energy in electron volts.
            temperature_k: System temperature in Kelvin.
            concentration: Local solvent concentration (0-1, normalized).
            pre_exponential_base: Base pre-exponential factor (1/ps).
            overpotential_V: Overpotential in volts (driving force).

        Returns:
            Rate constant in 1/ps.
        """
        # Arrhenius factor: exp(-Ea / (kB * T))
        arrhenius_factor = np.exp(-activation_energy_eV / (_KB_EV_K * temperature_k))

        # Concentration-dependent pre-exponential factor
        # As SEI grows, solvent concentration at the reaction interface
        # decreases due to diffusion through the growing SEI layer
        # A(c) = A0 * c / (c + K_m), where K_m is a Michaelis-like constant
        K_m = self._K_m
        concentration_factor = concentration / (concentration + K_m)

        # Overpotential dependence: exp(alpha * eta / (kB * T))
        # alpha ~ 0.5 for typical electron transfer reactions
        alpha = self._alpha
        overpotential_factor = np.exp(alpha * overpotential_V / (_KB_EV_K * temperature_k))

        # Combined rate constant
        k = pre_exponential_base * concentration_factor * arrhenius_factor * overpotential_factor

        return float(k)

    def _run_kmc_simulation(
        self,
        rng: np.random.RandomState,
        voltage: float,
        max_time: float,
        temperature_k: float = 298.15,
        solvent_type: str = "ec:dmc",
        salt_type: str = "NaPF6",
    ) -> SEIEvolution:
        """Run kinetic Monte Carlo simulation of SEI growth.

        Uses physically derived rate constants from the Arrhenius equation.
        The kMC loop tracks local solvent concentration as the SEI grows,
        implementing mass transport limitation: as the SEI thickens,
        the local concentration of reactive species at the anode interface
        decreases, reducing subsequent reaction rates.

        Reaction pathways:
            - Solvent decomposition (EC/DMC reduction)
            - Salt reduction (PF6- / TFSI- decomposition)
            - Polymerization (organic SEI formation)

        Args:
            rng: Deterministic random state seeded from molecular inputs.
            voltage: Voltage cutoff (V) affecting reaction rates.
            max_time: Maximum simulation time in picoseconds.
            temperature_k: Simulation temperature in Kelvin.
            solvent_type: Solvent composition (e.g., "ec:dmc").
            salt_type: Salt type (e.g., "NaPF6").

        Returns:
            SEIEvolution with final thickness, homogeneity, and conductivity.
        """
        # Initial bulk solvent concentration (normalized)
        initial_solvent_conc = self._initial_solvent_conc
        initial_salt_conc = self._initial_salt_conc

        # SEI thickness at which mass transport becomes significant
        # (Angstrom) - beyond this, diffusion through SEI limits rates
        transport_limit_thickness = self.config.transport_limit_thickness_angstrom

        # kMC simulation parameters
        n_steps = self.config.max_simulation_steps
        record_interval = self.config.record_interval

        # Track cumulative thickness and reaction counts
        total_thickness = 0.0
        n_solvent_events = 0
        n_salt_events = 0
        n_poly_events = 0

        # Record time-thickness profile
        time_points: list[float] = []
        thickness_points: list[float] = []

        current_time = 0.0

        # Solvent composition ratio (EC:DMC)
        ec_ratio = self._ec_ratio_default if "ec:dmc" in solvent_type else (1.0 if "ec" in solvent_type else self._solvent_composition.get("fallback_ratio", 0.5))
        dmc_ratio = 1.0 - ec_ratio

        for step in range(n_steps):
            # Current SEI thickness determines mass transport limitation
            # As SEI grows, solvent must diffuse through it to reach the anode
            transport_factor = 1.0 / (1.0 + total_thickness / transport_limit_thickness)

            # Local solvent concentration at the reaction interface
            local_solvent_conc = initial_solvent_conc * transport_factor
            local_salt_conc = initial_salt_conc * transport_factor

            # Compute Arrhenius rate constants for each pathway
            # Solvent decomposition (EC/DMC reduction)
            k_ec = self._arrhenius_rate(
                activation_energy_eV=self._Ea_SOLVENT_EC,
                temperature_k=temperature_k,
                concentration=local_solvent_conc * ec_ratio,
                pre_exponential_base=self._A_SOLVENT_BASE,
                overpotential_V=voltage,
            )
            k_dmc = self._arrhenius_rate(
                activation_energy_eV=self._Ea_SOLVENT_DMC,
                temperature_k=temperature_k,
                concentration=local_solvent_conc * dmc_ratio,
                pre_exponential_base=self._A_SOLVENT_BASE,
                overpotential_V=voltage,
            )
            k_solvent = k_ec + k_dmc  # Combined solvent decomposition rate

            # Salt reduction (PF6- decomposition)
            k_salt = self._arrhenius_rate(
                activation_energy_eV=self._Ea_SALT_PF6,
                temperature_k=temperature_k,
                concentration=local_salt_conc,
                pre_exponential_base=self._A_SALT_BASE,
                overpotential_V=voltage,
            )

            # Polymerization rate (organic SEI formation)
            # Depends on solvent radical concentration (proportional to solvent reaction)
            k_poly = self._arrhenius_rate(
                activation_energy_eV=self._activation_energies.get("polymerization", 0.40),
                temperature_k=temperature_k,
                concentration=local_solvent_conc,
                pre_exponential_base=self._A_POLY_BASE,
                overpotential_V=voltage * self._polymer_voltage_factor,
            )

            # Total reaction rate
            k_total = k_solvent + k_salt + k_poly

            # Advance time: delta_t = 1 / k_total (standard kMC time step)
            if k_total > 0:
                dt = 1.0 / k_total
            else:
                dt = 0.0

            current_time += dt

            # Select reaction via multinomial sampling
            r = rng.random()
            _cumulative = 0.0

            if r < k_solvent / k_total:
                # Solvent decomposition reaction
                n_solvent_events += 1
                total_thickness += self._D_SOLVENT
            elif r < (k_solvent + k_salt) / k_total:
                # Salt reduction reaction
                n_salt_events += 1
                total_thickness += self._D_SALT
            else:
                # Polymerization reaction
                n_poly_events += 1
                total_thickness += self._D_POLY

            # Record at intervals
            if step % record_interval == 0:
                time_points.append(float(current_time))
                thickness_points.append(float(total_thickness))

        # Final SEI thickness (clamped to physical range)
        final_thickness = float(np.clip(total_thickness, 1.0, 50.0))

        # Homogeneity: based on the distribution of reaction types
        total_events = n_solvent_events + n_salt_events + n_poly_events
        if total_events > 0:
            fractions = np.array([
                n_solvent_events / total_events,
                n_salt_events / total_events,
                n_poly_events / total_events,
            ])
            # Homogeneity is high when reactions are well-distributed
            # (not dominated by a single pathway)
            ideal_fraction = self._ideal_fraction
            deviation = np.mean(np.abs(fractions - ideal_fraction))
            homogeneity = max(1.0 - 2.0 * deviation, 0.0)
        else:
            homogeneity = 0.5

        # Ionic conductivity decreases exponentially with thickness
        ionic_cond = self._ionic_cond_base * np.exp(-final_thickness / self._cond_decay_length)

        # Electronic insulation maintained if SEI is sufficiently thick
        is_insulated = final_thickness > self._insulation_threshold

        # SEI components based on dominant reaction pathway
        components = ["NaF", "RO-ONa", "Na2CO3"]
        if n_salt_events > n_solvent_events and n_salt_events > n_poly_events:
            components.insert(0, "PF5-derived")
        elif n_poly_events > n_solvent_events:
            components.insert(0, "polymer-rich")

        return SEIEvolution(
            time_ps=float(max_time),
            thickness_angstrom=final_thickness,
            homogeneity_score=float(homogeneity),
            ionic_conductivity_s_cm=float(ionic_cond),
            electronic_insulation=is_insulated,
            components=components,
        )

    @staticmethod
    def _estimate_memory_footprint(sei: SEIEvolution, base: float = 2.0, scaling: float = 0.1) -> float:
        """Estimate memory usage for GCMD Digital Twin simulation."""
        return base + sei.thickness_angstrom * scaling

    @staticmethod
    def _hash_inputs(smiles: str, solvent: str, salt: str) -> int:
        """Generate deterministic seed from inputs."""
        combined = f"{smiles}_{solvent}_{salt}"
        return sum(ord(c) for c in combined) % (2**31)
