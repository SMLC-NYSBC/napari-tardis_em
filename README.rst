.. image:: resources/Tardis_logo_2.png
    :width: 512
    :align: center
    :target: https://smlc-nysbc.github.io/TARDIS/

========

.. image:: https://img.shields.io/github/v/release/smlc-nysbc/napari-tardis_em
        :target: https://img.shields.io/github/v/release/smlc-nysbc/tardis

.. image:: https://img.shields.io/badge/Join%20Our%20Community-Slack-blue
        :target: https://join.slack.com/t/tardis-em/shared_invite/zt-27jznfn9j-OplbV70KdKjkHsz5FcQQGg

.. image:: https://img.shields.io/github/downloads/smlc-nysbc/napari-tardis_em/total
        :target: https://img.shields.io/github/downloads/smlc-nysbc/tardis/total

.. image:: https://github.com/SMLC-NYSBC/tardis/actions/workflows/sphinx_documentation.yml/badge.svg
        :target: https://github.com/SMLC-NYSBC/tardis/actions/workflows/sphinx_documentation.yml

Napari plugin for napari-TARDIS-em
===========================

Napari [gen2] plugin for Cry-EM and Cryo-ET micrograph segmentation with TARDIS-em.

Citation
========

`DOI [BioRxiv] <http://doi.org/10.1101/2024.12.19.629196>`__

Kiewisz R. et.al. 2024. Accurate and fast segmentation of filaments and membranes in micrographs and tomograms with TARDIS.

`DOI [Microscopy and Microanalysis] <http://dx.doi.org/10.1093/micmic/ozad067.485>`__

Kiewisz R., Fabig G., Müller-Reichert T. Bepler T. 2023. Automated Segmentation of 3D Cytoskeletal Filaments from Electron Micrographs with TARDIS. Microscopy and Microanalysis 29(Supplement_1):970-972.

`Link: NeurIPS 2022 MLSB Workshop <https://www.mlsb.io/papers_2022/Membrane_and_microtubule_rapid_instance_segmentation_with_dimensionless_instance_segmentation_by_learning_graph_representations_of_point_clouds.pdf>`__

Kiewisz R., Bepler T. 2022. Membrane and microtubule rapid instance segmentation with dimensionless instance segmentation by learning graph representations of point clouds. Neurips 2022 - Machine Learning for Structural Biology Workshop.

Quick Start
===========

For more examples and advanced usage please find more details in our `Documentation <https://smlc-nysbc.github.io/TARDIS/>`__

0) Create new conda enviroment

.. code-block::

    conda create -n napari-tardis python=3.11
    conda activate napari-tardis

1) Install napari-TARDIS-em:

.. code-block:: bash

    pip install napari-tardis-em

3) Run Napari plugin

.. code-block:: bash

    napari