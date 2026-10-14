
import sys
from pathlib import Path

# Make project modules importable when running this test file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import calculate


def test_addition():
    assert calculate("125 * 8 + 40") == "125 * 8 + 40 = 1040"


def test_subtraction():
    assert calculate("100 - 35") == "100 - 35 = 65"


def test_division():
    assert calculate("81 / 9") == "81 / 9 = 9.0"


def test_division_by_zero():
    assert "division by zero" in calculate("10 / 0").lower()


def test_reject_unsupported_expression():
    result = calculate("1 + unknown")
    assert result.startswith("Calculation error:")
