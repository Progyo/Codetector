# Typedef: https://stackoverflow.com/questions/69446189/python-equivalent-for-typedef
# Either: https://medium.com/@rnesytov/using-either-monad-in-python-b6eac698dff5
# Generic: https://mypy.readthedocs.io/en/stable/generics.html

from typing import TypeVar, NewType
from oslash import Either
from .failure import Failure

### IMPORTANT -> REQUIRES typing.py patch !!!!
#PATCH: https://github.com/python/mypy/issues/3331#issuecomment-1416919806


T = TypeVar('T')
"""Generic type"""

Result = NewType[T]('Result',Either[Failure,T])
"""
Right result = success. Left result = failure.
"""

NoneResult = Result[None]
"""
Shorthand for Result[None].
"""