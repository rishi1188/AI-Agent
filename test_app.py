import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from app import UserManager, calculate_total


class FakeUser:
    def __init__(self, password):
        self.password = password

    def check_password(self, password):
        return self.password == password


class FakeDB:
    def __init__(self, users):
        self.users = users

    def find_user(self, username):
        return self.users.get(username)


def test_login_success():
    db = FakeDB({"alice": FakeUser("secret123")})
    manager = UserManager(db)
    assert manager.login("alice", "secret123") is True


def test_login_wrong_password():
    db = FakeDB({"alice": FakeUser("secret123")})
    manager = UserManager(db)
    assert manager.login("alice", "wrongpass") is False


def test_login_unknown_user():
    # This is the one that currently FAILS: login() crashes with an
    # AttributeError instead of returning False for an unknown user.
    db = FakeDB({})
    manager = UserManager(db)
    assert manager.login("nobody", "whatever") is False


def test_calculate_total():
    class Item:
        def __init__(self, price):
            self.price = price

    assert calculate_total([Item(10), Item(5), Item(2)]) == 17
