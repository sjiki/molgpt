"""pytest configuration loaded before any test module is imported.

Sets up two lightweight shims so tests can run in CI without the full
molsets/pomegranate stack or the RDKit SA_Score contribution directory:

  * moses / moses.utils  — only ``get_mol`` is used by the project code
    under test; we back it with the RDKit equivalent so the real logic
    is still exercised.
  * sascorer             — the SA_Score script is located in the RDKit
    contrib directory, which is not always present in pip-installed
    rdkit.  A fixed-return mock is sufficient for unit tests.
"""
import sys
import unittest.mock as mock

import matplotlib
matplotlib.use("Agg")

from rdkit import Chem  # noqa: E402 (must come after matplotlib.use)


# ── moses shim ─────────────────────────────────────────────────────────────
def _get_mol(smiles_or_mol):
    """Minimal get_mol compatible with the real moses implementation."""
    if isinstance(smiles_or_mol, str):
        return Chem.MolFromSmiles(smiles_or_mol) if smiles_or_mol else None
    return smiles_or_mol


_mock_moses_utils = mock.MagicMock()
_mock_moses_utils.get_mol.side_effect = _get_mol

_mock_moses = mock.MagicMock()
_mock_moses.utils = _mock_moses_utils

sys.modules.setdefault("moses", _mock_moses)
sys.modules.setdefault("moses.utils", _mock_moses_utils)

# ── sascorer shim ───────────────────────────────────────────────────────────
_mock_sascorer = mock.MagicMock()
_mock_sascorer.calculateScore.return_value = 2.5
sys.modules.setdefault("sascorer", _mock_sascorer)
