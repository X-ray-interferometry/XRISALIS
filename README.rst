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
 
The core scientific dependencies (``numpy``, ``astropy``, ``matplotlib``,
``pandas``, ``scipy`` and ``pillow``) are installed automatically and do not
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
