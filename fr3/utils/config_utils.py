"""Helpers for adapting dictionaries to attribute-based configuration objects."""


class DataClass:
    def __init__(self, arg_dict):
        self.update(arg_dict)

    def update(self, arg_dict):
        for key, value in arg_dict.items():
            setattr(self, key, value)
