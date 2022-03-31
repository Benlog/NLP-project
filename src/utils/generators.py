from random import choice, randint
from string import ascii_lowercase

from essential_generators import DocumentGenerator


def random_sentences():
    gen = DocumentGenerator()
    while True:
        yield gen.sentence()


def random_characters(length):
    letters = ascii_lowercase
    while True:
        words = " ".join(
            [
                "".join(choice(letters) for i in range(randint(1, 12)))
                for j in range(randint(3, length))
            ]
        )
        yield words[0].upper() + words[1:] + "."
