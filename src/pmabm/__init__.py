"""Agent-based model of Brenner's and Wood's theory of the transition to agrarian capitalism.

The formal specification this package implements is the Model Design section of
``paper/main.tex``; module and function docstrings cite the subsection each mechanism
comes from.
"""

from .config import ENGLAND, FRANCE, Params

__all__ = ["Params", "ENGLAND", "FRANCE"]
__version__ = "0.1.0"
