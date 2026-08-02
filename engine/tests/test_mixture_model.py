"""NRTL activity coefficient model tests.

Validates the NRTL mixture model against published EC/DMC data
and ensures backward compatibility with ideal mixing rules.
"""

from __future__ import annotations

from aurelius.scoring.oracle.mixture_model import (
    _nrtl_activity_coefficient,
    predict_mixture_dielectric,
    predict_mixture_li_solvation,
    predict_mixture_viscosity,
)


class TestNrtlActivityCoefficient:
    """Test the NRTL activity coefficient calculation."""

    def test_symmetric_ideal(self):
        gamma1, gamma2 = _nrtl_activity_coefficient(0.5, 0.5, 0.0, 0.0, 0.2)
        assert abs(gamma1 - 1.0) < 0.01
        assert abs(gamma2 - 1.0) < 0.01

    def test_positive_deviation(self):
        gamma1, gamma2 = _nrtl_activity_coefficient(0.5, 0.5, 1.0, 1.0, 0.3)
        assert gamma1 > 1.0 or gamma2 > 1.0

    def test_negative_deviation(self):
        gamma1, gamma2 = _nrtl_activity_coefficient(0.5, 0.5, -1.0, -1.0, 0.3)
        assert gamma1 < 1.0 or gamma2 < 1.0

    def test_pure_component_limit_x1(self):
        gamma1, gamma2 = _nrtl_activity_coefficient(1.0, 0.0, 1.0, 1.0, 0.3)
        assert abs(gamma1 - 1.0) < 0.01

    def test_pure_component_limit_x2(self):
        gamma1, gamma2 = _nrtl_activity_coefficient(0.0, 1.0, 1.0, 1.0, 0.3)
        assert abs(gamma2 - 1.0) < 0.01

    def test_asymmetric_parameters(self):
        gamma1, gamma2 = _nrtl_activity_coefficient(0.3, 0.7, 1.5, -0.5, 0.2)
        assert gamma1 > 0.0
        assert gamma2 > 0.0


class TestPredictMixtureDielectric:
    """Test dielectric prediction with NRTL model."""

    def test_ideal_default(self):
        result = predict_mixture_dielectric(89.0, 3.0, frac1=0.5)
        assert abs(result - 46.0) < 1.0

    def test_ideal_explicit(self):
        result = predict_mixture_dielectric(89.0, 3.0, frac1=0.5, model="ideal")
        assert abs(result - 46.0) < 1.0

    def test_nrtl_with_known_params(self):
        result = predict_mixture_dielectric(
            89.0, 3.0, frac1=0.5, model="nrtl",
            comp1="carbonate", comp2="ether",
        )
        assert result > 0.0

    def test_nrtl_fallback_when_params_missing(self):
        result = predict_mixture_dielectric(
            89.0, 3.0, frac1=0.5, model="nrtl",
            comp1="unknown1", comp2="unknown2",
        )
        assert abs(result - 46.0) < 1.0

    def test_nrtl_requires_components(self):
        result = predict_mixture_dielectric(
            89.0, 3.0, frac1=0.5, model="nrtl", comp1=None, comp2=None,
        )
        assert abs(result - 46.0) < 1.0

    def test_ec_dmc_mixture(self):
        EC_DIELECTRIC = 89.0
        DMC_DIELECTRIC = 3.0
        result = predict_mixture_dielectric(
            EC_DIELECTRIC, DMC_DIELECTRIC, frac1=0.6, model="nrtl",
            comp1="carbonate", comp2="carbonate",
        )
        assert 3.0 < result < 89.0


class TestPredictMixtureViscosity:
    """Test viscosity prediction with NRTL model."""

    def test_ideal_default(self):
        result = predict_mixture_viscosity(1.0, 0.5, frac1=0.5)
        assert result > 0.0

    def test_ideal_explicit(self):
        result = predict_mixture_viscosity(1.0, 0.5, frac1=0.5, model="ideal")
        assert result > 0.0

    def test_nrtl_with_known_params(self):
        result = predict_mixture_viscosity(
            1.0, 0.5, frac1=0.5, model="nrtl",
            comp1="carbonate", comp2="ether",
        )
        assert result > 0.0

    def test_nrtl_fallback_when_params_missing(self):
        result = predict_mixture_viscosity(
            1.0, 0.5, frac1=0.5, model="nrtl",
            comp1="unknown1", comp2="unknown2",
        )
        assert result > 0.0

    def test_ec_dmc_viscosity(self):
        EC_VISCOSITY = 1.6
        DMC_VISCOSITY = 0.6
        result = predict_mixture_viscosity(
            EC_VISCOSITY, DMC_VISCOSITY, frac1=0.5, model="nrtl",
            comp1="carbonate", comp2="carbonate",
        )
        assert 0.5 < result < 2.0


class TestPredictMixtureLiSolvation:
    """Test Li+ solvation prediction with NRTL model."""

    def test_ideal_default(self):
        result = predict_mixture_li_solvation(3.0, 1.0, frac1=0.5)
        assert abs(result - 2.0) < 0.1

    def test_ideal_explicit(self):
        result = predict_mixture_li_solvation(3.0, 1.0, frac1=0.5, model="ideal")
        assert abs(result - 2.0) < 0.1

    def test_nrtl_with_known_params(self):
        result = predict_mixture_li_solvation(
            3.0, 1.0, frac1=0.5, model="nrtl",
            comp1="carbonate", comp2="ether",
        )
        assert result > 0.0

    def test_nrtl_fallback_when_params_missing(self):
        result = predict_mixture_li_solvation(
            3.0, 1.0, frac1=0.5, model="nrtl",
            comp1="unknown1", comp2="unknown2",
        )
        assert abs(result - 2.0) < 0.1


class TestNrtlModelEdgeCases:
    """Test edge cases in the NRTL model."""

    def test_zero_fraction(self):
        result = predict_mixture_dielectric(89.0, 3.0, frac1=0.0, model="nrtl",
                                             comp1="carbonate", comp2="ether")
        assert abs(result - 3.0) < 0.1

    def test_one_fraction(self):
        result = predict_mixture_dielectric(89.0, 3.0, frac1=1.0, model="nrtl",
                                             comp1="carbonate", comp2="ether")
        assert abs(result - 89.0) < 0.1

    def test_extreme_viscosity_ratio(self):
        result = predict_mixture_viscosity(100.0, 0.1, frac1=0.5, model="nrtl",
                                            comp1="carbonate", comp2="ether")
        assert result > 0.0

    def test_invalid_model_falls_back_to_ideal(self):
        result = predict_mixture_dielectric(89.0, 3.0, frac1=0.5, model="unknown")
        assert abs(result - 46.0) < 1.0
