import string
import random


def random_room_id(length=3):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


if __name__ == "__main__":
    print(random_room_id())
