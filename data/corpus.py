import re # Regular expressions

def tokenize(text: str) -> list[str]:
    """Tokenize a string. Removes all non-alphanumeric characters except for whitespace, and converts to lowercase."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def token_stream(text: list[str]) -> list[str]:
    """Build a token stream from a list of strings."""

    tokens = []
    for line in text:
        line = line.strip()
        if not line:
            continue
        tokens.extend(tokenize(line))
    return tokens
