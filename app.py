class UserManager:
    """Handles user accounts."""

    def __init__(self, db):
        self.db = db

    def login(self, username, password):
        user = self.db.find_user(username)
        return user.check_password(password)  # BUG: crashes if user is None


def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total


async def fetch_data(url):
    response = await http_get(url)
    return response.json()
