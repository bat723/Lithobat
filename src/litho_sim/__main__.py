"""``python -m litho_sim`` — dispatches to the CLI.

The call is guarded: ``python -m`` sets ``__name__`` to ``"__main__"`` here,
while a plain import (an old console script pinned to
``litho_sim.__main__:main`` does exactly that) must not run the command as a
side effect — that used to execute every command twice.
"""

from litho_sim.cli import main

if __name__ == "__main__":
    main()
