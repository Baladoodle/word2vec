import re # Regular expressions

def tokenize(text: str, keep_punct: bool = True) -> list[str]:
    """Tokenize a string. Optionally keep punctuation; always lowercase."""

    text = text.lower()
    if keep_punct:
        return text.split()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def token_stream(lines: list[str], keep_punct: bool = True) -> list[str]:
    """Build a token stream from a list of strings."""

    tokens: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens.extend(tokenize(line, keep_punct=keep_punct))
    return tokens
