Guide
=====

High level overview
-------------------

The goal of this package is to provide reproducible and tested code to register (align) high-resolution zoom scans with lower-resolution overview scans.
It does this using the registration framework of ``SimpleITK``.

Within the ``SimpleITK`` framework registration maps points from one image to another image (or in our case one volume to another volume - to keep it simple we'll just use image from now on).
In ``SimpleITK`` language, we are looking for a transformation from a **fixed image** coordinate system to a **moving image** coordinate system.
Because we are looking for a transform to map the zoom to overview, we will set the fixed image to be the zoom and moving image to be the full organ.

The registration is done in several steps, described in the following sections.

Manually finding a common point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to manually find a common point between the two datasets being registered.
This provides a starting point for the registration code.

Initial rotational registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The first step of the automated pipeline finds the approximate relative rotation about the z-axis between the two datasets.
It does this by fixing the centre of rotation at the common point identified in both datasets, and then:

1. Scanning through the whole 0 - 360 degree angle range at a resolution of 2 degrees, and identifying the best match.
2. Scanning in a range of +/- 2.5 degrees about this best match at a resolution of 0.1 degrees to identify a more accurate best match.

Final full registration
~~~~~~~~~~~~~~~~~~~~~~~
The second and final step of the automated pipeline uses a rigid transform that allows for variation in translation (x, y, z compoonents), a rotation around the z-axis (one component), and a scale factor.
The scaling factor is varied to take into account uncertainties in the resolution of each dataset.
