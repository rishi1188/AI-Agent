import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import is_senior_discount_eligible

def test_exactly_65_is_eligible():
    assert is_senior_discount_eligible(65) is True

def test_under_65_not_eligible():
    assert is_senior_discount_eligible(40) is False

def test_over_65_is_eligible():
    assert is_senior_discount_eligible(70) is True
