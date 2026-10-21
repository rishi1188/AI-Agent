import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import get_or_default

def test_explicit_zero_is_kept():
    assert get_or_default(0, 99) == 0

def test_none_uses_default():
    assert get_or_default(None, 99) == 99

def test_normal_value_is_kept():
    assert get_or_default("hello", "default") == "hello"
