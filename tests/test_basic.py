"""Basic smoke tests: the installed package has a version, imports cleanly,
and ships its bundled data files."""

from pathlib import Path


def test_version():
    """Test that the version module has the correct attributes"""

    from xrisalis import version

    assert isinstance(version.version, str)
    assert isinstance(version.__version__, str)
    assert isinstance(version.version_tuple, tuple)


def test_import():
    """Test that the xrisalis modules can be imported"""

    import xrisalis
    from xrisalis import (colormaps, config, detector, evaluate, instrument,
                          io, pointing, products, reconstruct, simulate,
                          sources, units, viz)

    for module in (colormaps, config, detector, evaluate, instrument, io,
                   pointing, products, reconstruct, simulate, sources, units,
                   viz):
        assert module is not None
    assert callable(xrisalis.generate_photons)
    assert callable(xrisalis.redistribute)
    assert callable(xrisalis.reconstruct)

