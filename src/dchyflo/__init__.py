"""D-CHyFLO: carbon-aware hydropower–floating-PV modelling tools."""

from .curves import PhysicalCurves
from .dispatch import fixed_noncooperative_dispatch, optimize_dekadal_dispatch
from .fpv import simulate_unit_fpv
from .historical_reconciliation import reconcile_historical_operation
from .hydropower import calibrate_output_coefficient, evaluate_hydropower
from .reservoir import replay_water_balance
from .state_model import optimize_energy_aware_state

__version__ = "0.1.0"

__all__ = [
    "PhysicalCurves",
    "calibrate_output_coefficient",
    "evaluate_hydropower",
    "fixed_noncooperative_dispatch",
    "optimize_dekadal_dispatch",
    "optimize_energy_aware_state",
    "reconcile_historical_operation",
    "replay_water_balance",
    "simulate_unit_fpv",
]
