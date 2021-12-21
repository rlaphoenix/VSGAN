Building
========

This part of the documentation covers working with the Source Code of VSGAN, or it's documentation.

Preparing your Environment
--------------------------

Requirements
^^^^^^^^^^^^

.. _Python: https://python.org
.. _Pip: https://pip.pypa.io
.. _PEP-517: https://python.org/dev/peps/pep-0517
.. _Poetry: https://python-poetry.org/docs/#installation

- Python_ on a supported version, see ``pyproject.toml``'s Python dependency line.
- Pip_ on `v19.0 <https://pip.pypa.io/en/stable/news/#v19-0>`_ or newer (Pip with PEP-517_ support).
- Poetry_ on the latest version.

Configuring Poetry
^^^^^^^^^^^^^^^^^^

I recommend doing ``poetry config virtualenvs.in-project true`` to simplify the virtual-env creation
process. This also helps solve issues related to unnecessary venv re-creations, duplicate files,
and caching.

Getting the Source Code
^^^^^^^^^^^^^^^^^^^^^^^

Simply download the source code any way you wish, I recommend with git.
You can use `git checkout` to go to any point of the code, switch branches,
or make new ones.

.. code-block:: shell

  $ git clone https://github.com/rlaphoenix/VSGAN
  $ cd VSGAN

The rest of this document will assume you have a terminal at the folder of source code.

Installing Dependencies
^^^^^^^^^^^^^^^^^^^^^^^

As well as specific dev-environment dependencies which we have just gone through, you also need to
install the normal dependencies that VSGAN needs.

1. Install all of the :ref:`Software Requirements` from the Installation guide.
2. Install the Python-based dependencies with ``poetry install``.
3. If you want to build the docs locally, install the extra dependencies with
   ``poetry install -E docs``.

Your environment is now prepared, you can now go ahead and work on the Source Code or Documentation.

Installing from Source Code
---------------------------

.. warning::

  There are some caveats when installing from Source Code:

  1. Source code may have changes that are not yet tested or stable, and may have regressions.
  2. Only install from source code if you have a reason. E.g., to test changes.

If you wish to install from Source Code to your machine, outside of the poetry virtual-env that
is used for development:

.. code-block:: shell

    $ git clone https://github.com/rlaphoenix/VSGAN
    $ cd VSGAN
    $ python -m pip install --user .

You may also follow the steps below to build a distribution wheel which you can then use instead,
which has the added benefit of being shareable for quick installation.

However, if you *would* like to install it within the development virtual-env the ``poetry install``
call made earlier is done in editable mode. Meaning any change you make and then save (with some
exceptions) will be immediately reflected in the installed site-packages directory of the venv.

Building distribution files
---------------------------

Distribution files are builds (wheel) or source-code blobs (sdist) for quick and easy sharing and
installation with the changes made intact. These wheel files are the same ones used when installing
VSGAN and other software via ``pip``.

To make a distribution file. E.g., for a new release version. It's as simple as one call,
``poetry build``. This will build a wheel and sdist into the directory ``/dist``.

The built wheel can then be installed using pip, e.g.,:

.. code-block:: shell

  $ pip install dist/vsgan-1.4.0-py3-none-any.whl

Building the Documentation
--------------------------

You must have the extra documentation dependencies installed, ``poetry install -E docs``.

Working with the documentation from there can be done in many ways, like with sphinx-autobuild to
have it auto-reload changes, or with Visual Studio Code which can have a auto-reloading preview
on the right as you work on the documentation. Its how I've written what you are reading right now!

If you simply want to just build it to a HTML directory that you can then view in your browser
locally:

.. code-block:: powershell

  $ cd docs
  $ .\make html

The built documentation will then be in ``/docs/_build``, with the ``index.html`` at
``/docs/_build/html/index.html``.
