========
XRISALIS
========

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
X-Ray Interferometry Simulation and AnaLysIs Software
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

XRIsalis is a software package designed to provide end-to-end simulations of X-ray interferometric observations. It provides functionality to read in source models that are compatible with the SIMPUT standard, sample photons based on the source model and provided instrumental files, generate raw event lists, and finally produce images and spectra from the simulated data. 

Installation
------------
 
XRIsalis is not yet available on PyPI, so it is installed directly from the
source repository. The steps below cover the common cases.
 
Requirements
~~~~~~~~~~~~
 
* **Python 3.11 or newer**
 
The core scientific dependencies (``numpy``, ``astropy``, ``matplotlib``, ``scipy``, ``pillow`` and
`xrimodel <https://github.com/X-ray-interferometry/xrimodel>`_) are installed automatically and do not
need to be set up by hand. To run the example notebook provided in the repository, you will also need ``jupyter`` and ``ipywidgets``.
 
1. Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~~
 
.. code-block:: bash
 
   git clone https://github.com/X-ray-interferometry/XRIsalis.git
   cd XRIsalis
 
.. note::
 
   Clone the repository with ``git`` rather than downloading a ZIP archive.
   XRISALIS derives its version number from the git history via
   ``setuptools_scm``; a source tree without git metadata will fall back to a
   placeholder version.
 
2. Create an isolated environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
Using a dedicated environment keeps XRISALIS and its dependencies separate from
your system Python. Either ``venv`` or ``conda`` works.
 
Using ``venv``:
 
.. code-block:: bash
 
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
 
Using ``conda``:
 
.. code-block:: bash
 
   conda create -n xrisalis python=3.11
   conda activate xrisalis
 
3. Install the package
~~~~~~~~~~~~~~~~~~~~~~~~
 
For most users, a standard install is sufficient:
 
.. code-block:: bash
 
   pip install .
 
If you plan to modify the XRISALIS source code, install it in **editable**
mode so that your changes take effect without reinstalling:
 
.. code-block:: bash
 
   pip install -e .
 
To also pull in the development tools (e.g. the ``ruff`` linter):
 
.. code-block:: bash
 
   pip install -e ".[dev]"
 
4. Verify the installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
Check that the package imports and reports a version:
 
.. code-block:: bash
 
   python -c "import xrisalis; from xrisalis import version; print(xrisalis.version.__version__)"

xrimodel source models
----------------------

Source models built with `xrimodel <https://github.com/X-ray-interferometry/xrimodel>`_
(a spectrum per pixel, adaptive grids, shared light curves and power spectra, per-pixel
light-travel delays, 3-D binaries) are used like any other source. ``load_source``
recognises xrimodel files, and they combine with the built-in sources:

.. code-block:: python

   import xrisalis as xs

   src = xs.load_source("cygx1.fits")             # an xrimodel file -> XrimodelSource
   src = src + xs.point_source(0.0, 1e-7, spectrum)
   photons = xs.generate_photons(src, arf, 1e4, seed=1)
   events = xs.redistribute(photons, instrument, rmf, seed=2)

xrimodel draws the photons itself and assigns them to baselines by thinning against the
ARF. Its tangent-plane offsets (uas, x East, y North) are returned as absolute RA/Dec
around the model's reference position, ``src.reference_deg``.

``xrisalis.recipes`` runs the whole chain as a real observation would be analysed, and
gives what to compare the result with (the model convolved with the dirty beam of a point
source with the model's spectrum, reconstructed the same way):

.. code-block:: python

   from xrisalis import recipes

   obs = recipes.observe(model, 2e4, fov_arcsec=0.0012)   # detected energies, deconvolved
   beam = recipes.dirty_beam(obs, 2e4)                    # unit peak, integrates to ~0
   setup = recipes.Setup.from_files("my_arf.fits", "my_rmf.fits")
   obs2 = recipes.observe(model, 2e4, fov_arcsec=0.0012, setup=setup)

The default ``setup`` is the bundled 10-baseline ARF and RMF, rolling half a turn.

Sky coordinates
---------------

Sources are placed at absolute RA/Dec. ``generate_photons`` records a nominal pointing
(``pointing=(ra, dec)``; default: the source's own position), and every position is
projected onto the tangent plane there (gnomonic; ``xrisalis.coords``), so offsets are
true angles at any declination and across RA = 0. The attitude stream's ``(pitch, yaw)``
are North and East offsets from the nominal pointing: ``(0, 0)`` stares at it, and
``xrisalis.sky_offset_deg(ra, dec, ra0, dec0)`` gives the offset that points at another
position. ``reconstruct(phase_center=(ra, dec))`` and ``evaluate(..., center_deg=...)``
take absolute coordinates.

Each photon sees the attitude (pitch, yaw, roll) interpolated at its own arrival time
(``xrisalis.pointing.attitude_at``), both in ``redistribute`` and in the per-event roll
and boresight used by ``reconstruct``.

