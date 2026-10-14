
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import route_question


def classify(question):
    result = route_question({"question": question})
    return result["route"]


def test_calculation_route():
    assert classify("Calculate 125 * 8 + 40") == "calculator"


def test_report_route():
    assert classify("Generate a report about the uploaded document") == "report"


def test_document_search_route():
    assert classify("According to the uploaded PDF, what is the main finding?") == "document_search"


def test_general_question_route():
    assert classify("Explain the difference between AI and machine learning") == "general"
