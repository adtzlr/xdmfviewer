from __future__ import annotations

from xdmfviewer.version import __version__


def test_version_string_is_defined() -> None:
    assert isinstance(__version__, str)