import re
import unicodedata

# Minimal map for common Greek letters seen in names
_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "κ": "kappa", "λ": "lambda", "μ": "mu",
    "ν": "nu", "ω": "omega"
}

_ROMAN_TO_INT = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
    "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10
}




def normalize_name(name: str) -> str:
   
    s = unicodedata.normalize("NFKC", name)

    # Greek letters to latin words
    s = "".join(_GREEK.get(ch, ch) for ch in s)

    # Strip diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Lowercase
    s = s.lower()
    

    # Normalize quotes/dashes
    s = (s.replace("’", "'").replace("`", "'").replace("´", "'")
           .replace("–", "-").replace("—", "-").replace("−", "-"))

    # Drop possessive 's (e.g., "joubert's" -> "joubert")
    s = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", s)

    # Replace non-word punctuation with spaces (keep hyphen for one more step)
    s = re.sub(r"[^\w\s-]", " ", s)

    # Hyphens -> spaces
    s = s.replace("-", " ")

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # 'type <roman>' -> 'type <arabic>'
    def _roman_sub(m):
        r = m.group(1).lower()
        return f"type {_ROMAN_TO_INT.get(r, r)}"
    s = re.sub(r"\btype\s+(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", _roman_sub, s, flags=re.I)

    return s