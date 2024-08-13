hipct-reg
=========

Code for registering high-resolution HiP-CT region of interest datasets to low-resolution HiP-CT full-organ datasets.

This package contains the image registration code.
Helpers for keeping an inventory of registered data and doing the registrations for HiP-CT data are in the (private) ``hipct-data-tools`` repository.
This package designed to be used internally by the HiP-CT team, but publicly released so our registrations are reproducible and in case anyone else finds the code helpful.

Contents
--------

- :doc:`auto_examples/tutorial`: A lower level tutorial showing how to use the registratino code from Python.
- :doc:`guide`: An explanation of how the registration pipeline works.
- :doc:`api`: API reference for different bits of the registration pipeline.

.. toctree::
   :maxdepth: 2
   :hidden:

   auto_examples/tutorial
   guide
   api
