"""English rendering of the Spanish sample descriptions of the source study.

The characterisation campaign of Blasco et al. (2021) lists every compound by a
free-text Spanish description of the batch as it was delivered --- how many
plates or pairs of soles arrived, in what colour, under what trade designation.
Those strings are the natural sample identifiers and are used as such in the
figures, but a manuscript written in English cannot carry them untranslated.

The descriptions follow a regular pattern,

    <quantity> <form> [<material>] <colour> <grade ...> [<treatment>]

over a closed vocabulary of eight colours, three container forms and three
treatments, with everything the parser does not recognise --- trade names,
hardness grades, internal codes --- carried through verbatim, since those are
proper nouns and must not be translated. Rendering the parsed fields in English
adjective-noun order gives idiomatic text rather than the word-by-word
substitution a lookup table would produce.

One class of token is dropped rather than carried through: the brand of the
footwear company that supplied a batch, which names a customer and not a
material. The source ``description`` keeps the study's text unaltered.

The translation is deliberately mechanical and total: :func:`translate` raises
on any Spanish word it does not know, so that a description added later cannot
silently reach a figure in the wrong language.
"""

from __future__ import annotations

import re

__all__ = ["translate", "UnknownTerm"]


class UnknownTerm(ValueError):
    """A description contains a Spanish word outside the closed vocabulary."""


# Container in which the batch was supplied.
FORMS: dict[str, tuple[str, str]] = {
    "plancha": ("sheet", "sheets"),
    "planchas": ("sheet", "sheets"),
    "par": ("pair", "pairs"),
    "pares": ("pair", "pairs"),
    "muestra": ("sample", "samples"),
    "muestras": ("sample", "samples"),
}

COLOURS: dict[str, str] = {
    "negro": "black",
    "marron": "brown",
    "amarillo": "yellow",
    "caramelo": "tan",
    "crudo": "natural",
    "transparente": "clear",
    "gris": "grey",
    "blanco": "white",
}

# "C/" abbreviates "color"; "C/gris" is a grey-coloured batch.
COLOUR_PREFIX = re.compile(r"^c/(?P<colour>\w+)$", re.IGNORECASE)

TREATMENTS: dict[str, str] = {
    "lavado": "washed",
    "lavados": "washed",
    "relleno": "filled",
}

# Material names and family acronyms. The acronyms are carried through
# unchanged but are recognised here so that they take the adjective-noun
# position ("black TPU sheets") rather than being pushed into the designation.
MATERIALS: dict[str, str] = {
    "caucho": "rubber",
    "latex": "latex",
    "eva": "EVA",
    "tpu": "TPU",
    "tr": "TR",
    "pur": "PUR",
    "pvc": "PVC",
    "pu": "PU",
}

# Function words carrying no information once the fields are reordered.
STOPWORDS = {"de", "del", "la", "el", "los", "las", "y"}

# Brands of the footwear companies that supplied the batches, as opposed to the
# grade designations of the compounds themselves. They identify a customer
# rather than a material and are dropped from the published rendering; the
# ``description`` column keeps the source study's text unaltered.
DROP_BRANDS = frozenset({"pikolinos"})

# Multi-word constructions, matched before the token pass.
PHRASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsin\s+lavar\b", re.IGNORECASE), "@unwashed"),
    (re.compile(r"\bmarcada?\s+como\b", re.IGNORECASE), "@labelled"),
]

_ACCENTS = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")


def _fold(word: str) -> str:
    return word.translate(_ACCENTS).lower()


def translate(description: str) -> str:
    """Render a Spanish batch description in English.

    Parameters
    ----------
    description
        The free-text description as it appears in the source study, e.g.
        ``"5 planchas PUR negro BK sin lavar"``.

    Returns
    -------
    str
        The English rendering, e.g. ``"5 black PUR sheets, BK, unwashed"``.

    Raises
    ------
    UnknownTerm
        If a lower-case word is neither in the vocabulary above nor recognisable
        as a designation (which are kept verbatim). Designations are detected by
        case and punctuation: a token containing an upper-case letter, a digit
        or a slash is a trade name, a hardness grade or an internal code.
    """
    text = description.strip()
    for pattern, marker in PHRASES:
        text = pattern.sub(marker, text)

    quantity: str | None = None
    form: tuple[str, str] | None = None
    colour: str | None = None
    material: str | None = None
    labelled = False
    treatments: list[str] = []
    designation: list[str] = []

    for token in text.split():
        folded = _fold(token)

        if quantity is None and designation == [] and token.isdigit():
            quantity = token
            continue
        if folded == "@unwashed":
            treatments.append("unwashed")
            continue
        if folded == "@labelled":
            labelled = True
            continue
        if folded in STOPWORDS:
            continue
        if folded in FORMS:
            form = FORMS[folded]
            continue
        if folded in TREATMENTS:
            treatments.append(TREATMENTS[folded])
            continue
        if folded in COLOURS and colour is None:
            colour = COLOURS[folded]
            continue
        # Everything after "marcada como" is a quoted trade designation and is
        # reproduced verbatim, so no material is extracted from it.
        if folded in MATERIALS and material is None and not labelled:
            material = MATERIALS[folded]
            continue

        match = COLOUR_PREFIX.match(folded)
        if match and colour is None:
            key = match.group("colour")
            if key not in COLOURS:
                raise UnknownTerm(f"unknown colour {key!r} in {description!r}")
            colour = COLOURS[key]
            continue

        if folded in DROP_BRANDS:
            continue

        # Anything left must look like a designation rather than a Spanish word.
        if token.islower() and token.isalpha():
            raise UnknownTerm(f"untranslated word {token!r} in {description!r}")
        designation.append(token)

    plural = quantity is not None and quantity != "1"
    head = [part for part in (quantity, colour, material) if part]
    if form is not None:
        head.append(form[1] if plural else form[0])

    out = " ".join(head)
    if designation:
        joined = " ".join(designation)
        out += f", labelled {joined}" if labelled else f", {joined}"
    if treatments:
        out += ", " + " and ".join(treatments)
    return out
