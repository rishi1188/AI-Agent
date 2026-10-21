import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import classify_number

def test_positive():
    assert classify_number(5) == "positive"

def test_negative():
    assert classify_number(-3) == "negative"

def test_zero():
    assert classify_number(0) == "zero"
