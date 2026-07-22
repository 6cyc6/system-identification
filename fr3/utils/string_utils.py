"""Small string manipulation helpers."""


def concatenate_strings(strings):
    return "".join(strings)


def concanate_strings(lst_strings):
    """Backward-compatible alias for the original misspelled function name."""
    return concatenate_strings(lst_strings)
