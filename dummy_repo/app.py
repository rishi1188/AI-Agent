class UserManager:
    """Handles user accounts."""

    def __init__(self, db):
        self.db = db

    def login(self, username, password):
        user = self.db.find_user(username)
        if user and user.check_password(password):
            return True
        return False


def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total


async def fetch_data(url):
    response = await http_get(url)
    return response.json()