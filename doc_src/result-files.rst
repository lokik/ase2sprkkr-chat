Result files, plots and NeXus export
====================================

Structured SPR-KKR result files
-------------------------------

ASE2SPRKKR includes parsers for several structured scientific
SPR-KKR outputs (such as DOS, BSF, SPC, JXC, RAT and SFN).
Plotting requires the optional ``plot`` dependencies; install them
if they are not already available::

   python -m pip install 'ase2sprkkr[plot]'

Then plot a recognized file::

   # Plot the result values.
   ase2sprkkr plot calculation.dos

   # Inspect which result-file formats are supported.
   ase2sprkkr info --output-file

   # Show the names of the values available in this file.
   ase2sprkkr plot calculation.dos --show-values

Available values depend on the format and installed package version.

Load a recognized structured file using ``ResultFile``:

.. code-block:: python

   from ase2sprkkr import ResultFile

   data = ResultFile.from_file("calculation.dos")

``ResultFile`` is the current public name; ``OutputFile`` is a
deprecated compatibility name. Result-file parsing is independent of
running a new calculation.

Loading related calculation outputs
-----------------------------------

Use :class:`ase2sprkkr.outputs.ResultLoader` to import multiple native
``.out`` files with a shared directory index and short-lived caches for
input parameters and potentials. The loader does not store GUI history or
search for unrelated file extensions.

.. code-block:: python

   from ase2sprkkr import ResultLoader, FileAssociation

   batch = ResultLoader.load_directory("/calculations/Fe")
   for result in batch.results:
       print(result.task_name, result.files["output"].path())

   # Recent outputs can share the same cache without a directory glob:
   loader = ResultLoader()
   recent = loader.load_files(["Fe_DOS.out", "Fe_SCF.out"])
   print(recent.failures)  # Individual bad outputs do not abort the batch.

Each result owns the association of its files. A file can be absent or
overwritten without losing the fact that it was named in the original
output or input. For example:

.. code-block:: python

   file = result.files["converged"]
   print(file.path(), file.association, file.association_evidence)
   print(file.recorded_version, file.observed_version)

``UNKNOWN`` means no reliable conclusion; ``LIKELY`` is heuristic;
``NOT_THIS_RESULT`` indicates that the file is attributed elsewhere
(e.g. a newer SCF using the same filename); ``VERIFIED`` requires a
matching historical content digest from the run manifest. Only the
result-file option holds these values; there is no parallel provenance
registry on ``TaskResult``.

The version observed at import protects against later rewrites, but
**does not prove** which past calculation produced those bytes.
``associated_with`` may identify the newer competing SCF stdout.
Use ``file.is_current()`` before operating on a file; cached results
are re-associated when reused. One ``ResultLoader`` instance is intended
for one import worker and is not thread-safe.

Grace .agr projects
-------------------

Grace ``.agr`` projects are supported separately from the structured
SPR-KKR file registry:

.. code-block:: python

   from ase2sprkkr.result_files import read_agr, read_plottable_file

   grace = read_agr("bands.agr")
   print(grace.datasets)
   data = read_plottable_file("bands.agr")

The latter helper dispatches by filename and handles both Grace
projects and recognized SPR-KKR result files.

To plot a Grace file or save a figure::

   ase2sprkkr plot bands.agr
   ase2sprkkr plot bands.agr -o bands.png

To list the values in a Grace file, use::

   ase2sprkkr plot bands.agr --show-values

The ``--value`` option selects a value to plot and can be repeated. SFN shape-function files additionally support
``--mesh``, ``--cell``, ``--periodic``, ``--shape``,
``--slice`` and ``--radial`` plotting modes.

ARPES SPC to NeXus/HDF5
-----------------------

The 4.0 series can export supported **ARPES .spc spectra** in NeXus
format. Install the separate optional dependencies::

   python -m pip install 'ase2sprkkr[nexus]'

Export from the command line::

   ase2sprkkr nexus arpes.spc
   ase2sprkkr nexus arpes.spc -o arpes.nxs -i arpes.inp

By default, the output has the same basename as the source and a
``.nxs`` suffix. The optional ``-i`` argument supplies the native
SPR-KKR input file for provenance; an unambiguous accompanying input
file can also be found automatically. The exporter retains available
axes, spectra and source settings.

From Python, use the same writer on a compatible SPC result:

.. code-block:: python

   from ase2sprkkr import ResultFile

   spectrum = ResultFile.from_file("arpes.spc", try_only="spc")
   spectrum.to_nexus("arpes.nxs", input_file="arpes.inp")

Missing experimental geometry or calibration cannot be reconstructed
from the SPC file. Intensities may therefore be labeled as arbitrary
units; this does **not** mean they have been normalized or calibrated.

For API help and the CLI syntax, see :doc:`cli`.
