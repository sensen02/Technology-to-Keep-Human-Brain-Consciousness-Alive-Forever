"""One place that knows how to write a legal XML comment.

WHY THIS IS A MODULE AND NOT A ONE-LINER AT EACH CALL SITE.

    XML forbids "--" inside a comment.  Several tools in this repository stamp a provenance
    comment into the models they generate, and a comment carrying an ordinary English phrase like
    "arm -- sign included -- the reference's" makes the WHOLE DOCUMENT unparseable.  MEASURED:
    `fruitfly_six_leg_standing.xml` failed under `xml.etree` with
    "not well-formed (invalid token): line 15, column 36", while `_load_mjcf`'s lenient reader
    accepted it happily -- so the model worked in the pipeline and broke for everything that
    parses XML properly.

    The obvious fix, `prov.replace("--", "-")` over the whole string, is WRONG for the same
    reason it is tempting: it also rewrites the `<!--` and `-->` DELIMITERS, producing `<!-` and
    `->`, i.e. a comment that is not a comment.  That is exactly what the first attempt at this
    fix did, and it was caught immediately by the same guard.  So the sanitiser works on the
    inner text only, and the delimiters are added afterwards.
"""
from __future__ import annotations

# an em dash reads the same in a comment and is legal XML
_DASH = "\u2014"


def xml_comment(inner: str) -> str:
    """Return `inner` wrapped in a comment that is guaranteed to be legal XML.

    `inner` must NOT contain the comment delimiters; they are added here, after sanitising.
    """
    body = inner.replace("--", _DASH)
    if body.endswith("-"):
        body += " "
    return "<!--" + body + "-->"


def sanitise_inner(inner: str) -> str:
    """The inner-text half of `xml_comment`, for callers that add their own delimiters."""
    body = inner.replace("--", _DASH)
    return body + " " if body.endswith("-") else body
