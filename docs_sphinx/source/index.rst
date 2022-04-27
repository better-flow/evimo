.. EVIMO documentation master file, created by
   sphinx-quickstart on Mon Apr 18 15:03:53 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EVIMO's documentation!
=================================

**evimo** is a toolkit for fusing data streams from multiple cameras (event-based, rgb or potentially any other kind of visual sensor) with a motion capture system in a AR-like fashion to automatically generate ground truth annotations for motion, depth and scene segmentation (both motion or semantic). The toolkit uses static 3D scans of the objects on the scene to achieve this goal - the objects and cameras are fitted with motion capture markers and the simulated ground truth is overlaid on the real data from sensors.

.. toctree::
   :maxdepth: 2
   :caption: Data

   ground-truth-format.md
   evimo2v2-inspect-sequence.md
   evimo-flow.md

.. toctree::
   :maxdepth: 2
   :caption: Generation

   docker-environment.md
   evimo2v2-generation.md
   offline-generation-tool.md

.. toctree::
   :maxdepth: 2
   :caption: Recording

   evimo-pipeline-setup.md
   raw-sequence-structure.md
   raw-sequence-inspection.md
   calibration-tools.md
   adding-a-new-object.md
