import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import add_to_cart

def test_carts_are_independent_between_calls():
    cart_a = add_to_cart("apple")
    cart_b = add_to_cart("banana")
    assert cart_b == ["banana"]

def test_explicit_cart_still_works():
    my_cart = []
    result = add_to_cart("milk", my_cart)
    assert result == ["milk"]
