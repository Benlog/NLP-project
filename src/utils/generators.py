from random import choice, randint
from string import ascii_lowercase

from essential_generators import DocumentGenerator
from faker import Faker


def random_fake_sentences(seed=None):
    fake = Faker()

    if seed:
        Faker.seed(seed)

    while True:
        return fake.sentance()


def random_sentences():
    gen = DocumentGenerator()
    while True:
        yield gen.sentence()


def random_characters(length):
    letters = ascii_lowercase
    words = " ".join(
        [
            "".join(choice(letters) for i in range(randint(1, 12)))
            for j in range(randint(3, length))
        ]
    )
    return words[0].upper() + words[1:] + "."


def random_characters_generator(min_length: int, max_length: int):
    while True:
        length = randint(min_length, max_length)
        yield(random_characters(length))
