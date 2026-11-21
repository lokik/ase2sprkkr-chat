Using ASE2SPRKKR
================

The ``SPRKKR`` calculator prepares and runs native SPR-KKR tasks from
ASE structures or existing potentials. If you have not yet run a
calculation, follow the :doc:`quickstart` first. For physics and
convergence settings, consult the
`SPR-KKR manual <https://sprkkr.org/manual/>`_.

Tasks and input parameters
--------------------------

The default task is ``SCF``. Other tasks include ``DOS``, ``ARPES``,
``SPLEED``, ``BSFEK``, ``BSFKK`` and ``JXC``; some less established
tasks are marked experimental. Inspect the installed version to find
task names and available options::

   ase2sprkkr info --task
   ase2sprkkr info --task=SCF
   ase2sprkkr info --task=SCF.ENERGY

A calculator can receive a task name, an input-file path or an
``InputParameters`` object. For example:

.. code-block:: python

   from ase.build import bulk
   from ase2sprkkr import InputParameters, SPRKKR

   parameters = InputParameters.for_task("SCF")
   parameters.ENERGY.NE = 5

   atoms = bulk("Al", "fcc", a=4.05)
   calculator = SPRKKR(atoms=atoms, input_parameters=parameters)
   result = calculator.calculate()

   print("SCF energy (eV):", result.energy)
   print("Converged:", result.converged)

You can change an option on the calculator with
``calculator.set({"ENERGY.NE": 6})``, or override it for one run with
``calculator.calculate(options={"ENERGY.NE": 6})``.
A section-qualified name avoids ambiguity when different sections have
options with the same name.

Interactive parameter editing
-----------------------------

Input-parameter defaults belong to the calculator and can be inspected
and changed before running a task. For example:

.. code-block:: python

   from ase.build import bulk
   from ase2sprkkr import SPRKKR

   calculator = SPRKKR(atoms=bulk("Al", "fcc", a=4.05),
                       input_parameters="SCF")
   calculator.set({"ENERGY.NE": 12})
   print(calculator.get("ENERGY.NE"))

   # The same option can be accessed through its section:
   calculator.input_parameters.ENERGY.NE = 16
   print(calculator.input_parameters.ENERGY.NE())

   # This changes the option for just one calculation:
   result = calculator.calculate(options={"ENERGY.NE": 24})

The calculator retains its default value of 16 after this one-time
override. To select a different task or replace all parameter defaults,
assign to ``calculator.input_parameters`` (for example, ``"DOS"``,
an ``InputParameters`` instance, or an input-file path). The
``input_parameters`` argument to ``calculate()`` can also select
a task for a single calculation without replacing the defaults.
Use ``ase2sprkkr info --task=TASK`` to discover valid settings.

To inspect an existing SPR-KKR input file without running a calculation:

.. code-block:: python

   from ase2sprkkr import InputParameters

   parameters = InputParameters.from_file("existing.inp")
   print(parameters.task_name)

Incomplete files can be inspected with
``InputParameters.from_file("existing.inp", allow_incomplete=True)``.
Input files do not contain all the physical data needed to run
a calculation; a potential or atoms must also be provided.
For task-definition internals and shared parameter schemas, see
:doc:`philosophy`.

Reusable input presets
----------------------

Presets store explicitly assigned input parameters in YAML. They are organized
in two scopes under the ASE2SPRKKR user configuration directory::

   presets/
     input_parameters/bsf/common.yaml
     task/bsfek/along_path.yaml
     task/bsfkk/fermi_surface.yaml

An ``input_parameters`` preset must be compatible with *every* task using
the named IPD. A ``task`` preset belongs to one concrete backend task.
Shared IPDs such as BSF (BSFEK/BSFKK) and JXC (JXC/DMI) are supported.
The directory name provides the definition or task identity.

Saving with a bare name automatically selects ``input_parameters`` scope
if the explicitly stored values are task-neutral; otherwise it selects
the current ``task``. Use a qualified name to require a particular scope::

   from ase2sprkkr import InputParameters

   ip = InputParameters.for_task("BSFEK")
   ip.ENERGY.NE = [400]
   ip.save_preset("along_path")               # task:bsfek.along_path
   ip.save_preset("task:bsfek.my_path")       # Explicit task scope

   ip = InputParameters.for_definition("BSF")
   ip.TAU.NKTAB = 500
   ip.save_preset("input_parameters:bsf.dense")

   ip.load_preset("task:bsfkk.fermi_surface")  # Replace and switch to BSFKK
   ip.load_preset("input_parameters:bsf.dense",
                  clear_first=False)          # Merge, preserve current task

Loading replaces calculation settings by default (``clear_first=True``);
merging (``clear_first=False``) preserves the current effective task.
Replacing with an IPD-neutral preset also preserves the current effective task.
Replacing with a task-specific preset can select that task as long as the
caller has not supplied a conflicting ``required_task``. A merge of a
different task-specific preset is rejected.

By default, loading retains the original dataset, ADSI suffix, potential
paths, embedding mode (CLUTYPE), and shape-function filenames; use
``load_structural_info=True`` to load these from the preset instead.
An omitted option or a YAML ``null`` leaves that value untouched when
merging. Explicit values, including ``false``, are applied in either mode.
For example, this preset enables symmetry by clearing ``NOSYM``::

   version: 1
   parameters:
     CONTROL:
       NOSYM: false

Manage presets from the command line::

   ase2sprkkr presets list
   ase2sprkkr presets list input_parameters:bsf
   ase2sprkkr presets list task:bsfkk
   ase2sprkkr presets show task:bsfkk.fermi_surface
   ase2sprkkr presets create task:bsfkk.fermi_surface previous.inp
   ase2sprkkr presets create dos.with_symmetry symmetry.yaml
   ase2sprkkr run sample.pot --preset task:bsfkk.fermi_surface

Use ``presets create IPD.NAME INPUT.inp`` to import an older native preset
into the YAML store. Native SPR-KKR input files still use ``.inp``.

The ``input_parameters:jxc.NAME`` and ``task:jxc.NAME`` forms distinguish
the JXC schema from the JXC task. Unqualified names are accepted only when
their scope and identity are unambiguous. Explicit ``run --task`` takes
precedence over a conflicting task-specific preset. Running a neutral
preset for a shared IPD without any task context requires an explicit
``--task``, rather than silently selecting a default mode. Command-line
``NAME=VALUE`` overrides are applied after loading a preset.

Site occupations and symmetry
-----------------------------

ASE structures can describe geometry, but SPR-KKR needs additional
site properties, including chemical occupations and atomic-type data.
``promote_ase_atoms()`` adds the ``sites`` interface to an ASE
``Atoms`` object. Promotion also happens when atoms are supplied to
the ``SPRKKR`` calculator.

Symmetry-equivalent sites can share a ``SiteType``. To change
**only one** site, first call ``break_symmetry()``; otherwise editing
a shared occupation raises an error:

.. code-block:: python

   from ase.build import bulk
   from ase2sprkkr import promote_ase_atoms

   atoms = promote_ase_atoms(
       bulk("LiCl", "rocksalt", a=5.64) * (2, 1, 1)
   )
   site = atoms.sites[3]  # one of the chlorine sites
   site.break_symmetry()
   site.occupation.set({"Cl": 0.5, "I": 0.5})

   # Adding Br at 20% reduces the existing fractions proportionally:
   site.occupation["Br"] = 0.2
   print(site.occupation["Br"])
   print(site.occupation.as_dict)

To modify **all** equivalent sites together, edit their shared
``site.site_type.occupation`` instead of breaking symmetry.
Occupation values are normalized; atomic types can be specified by
their chemical symbols or atomic numbers. Changes are synchronized to
ASE's ``atoms.info["occupancy"]`` representation using the
``spacegroup_kinds`` mapping, and existing ASE occupancy metadata is
recognized when atoms are promoted.

Different occupation fractions represent chemical disorder (such as
a substitutional alloy), not a larger explicitly ordered supercell.
The consequences of changing equivalence or converting a disordered
cell are discussed in :doc:`symmetry`.

Potentials and consecutive tasks
--------------------------------

An SPR-KKR ``.pot`` file contains atomic-site information and the
electronic potential used by native calculations. It is not a
pseudopotential library such as VASP's POTCAR.

Create a starting potential from an ASE structure, or load an existing
one:

.. code-block:: python

   from ase.build import bulk
   from ase2sprkkr import Potential

   potential = Potential.from_atoms(bulk("Al", "fcc", a=4.05))
   previous = Potential.from_file("existing.pot")

ASE's standard ``read()`` and ``write()`` functions support both
ordinary SPR-KKR potentials and native embedded-cluster ``CLUPOT`` files.
For clusters, supply the separate host potential explicitly: CLUPOT does
not contain the host cell or full host structure.

.. code-block:: python

   from ase.io import read, write
   from ase2sprkkr import Potential

   host = Potential.from_file("host.pot")
   cluster = read("embedded.clupot", format="SPRKKR", host=host)
   cluster.sites[0].embedding.displacement = (0.1, 0.0, 0.0)
   write("edited.clupot", cluster, format="SPRKKR")

For potential objects, the common ``Potential.from_file()`` entry point
accepts both native POTFIL and CLUPOT contents (including non-seekable
file streams) and returns a concrete ``PeriodicPotential`` or
``EmbeddedPotential``. Their own ``from_file()`` methods are
strict about the native format; neither accepts the other file type.
The historic ``ClusterPotential`` name remains an alias for
``EmbeddedPotential``. Parsing CLUPOT does not require a host; only
reconstructing embedded ASE atoms does.

The ``host`` argument may also be a path such as ``host="host.pot"``.
The same reader detects CLUPOT by its contents, even when the filename
ends in ``.pot``. Writing a zero-dimensional embedded structure selects
CLUPOT automatically; the host must remain attached to the atoms as
``atoms.info["sprkkr_embedding_host"]``. To examine a CLUPOT without
a host, load its sections directly using
``ClusterPotential.from_file("embedded.clupot")`` instead of ASE's
atoms reader.

When starting from atoms, ASE2SPRKKR prepares a potential containing
the site information needed for the native calculation. When starting
from a previous ``.pot`` file, the stored electronic state can be
reused. Input parameters specify the task and its controls; they do
**not** replace the structure and electronic data in the potential.
For that reason, loading an ``.inp`` file alone is insufficient
unless its referenced potential can also be found.

A calculator owns default atoms or a potential, along with default
input parameters. Explicit ``atoms``, ``potential`` and
``input_parameters`` arguments to ``calculate()`` select the
physical state and task for that run. Do not supply both an ``Atoms``
object and a ``Potential`` object to the calculator constructor;
the latter already contains the atomic structure. During preparation,
input parameters are copied before per-run options and generated paths
are inserted, but atoms and potential objects are mutable working
state: promoting atoms, adding empty spheres, or synchronizing the
potential may modify them. Once prepared, a run uses its own
materialized files.

A converged SCF potential can be used for a subsequent DOS calculation:

.. code-block:: python

   from ase.build import bulk
   from ase2sprkkr import SPRKKR

   scf = SPRKKR(atoms=bulk("Al", "fcc", a=4.05)).calculate()
   if not scf.converged:
       raise RuntimeError("SCF did not converge")

   dos = SPRKKR(input_parameters="DOS").calculate(
       potential=scf.potential_filename
   )
   print(dos.dos_filename)

Passing the potential explicitly identifies which SCF run supplies it.
Alternatively, the SCF result provides a calculator associated with
the new potential, suitable for the next task:

.. code-block:: python

   dos = scf.calculator.calculate(input_parameters="DOS")

A successful synchronous SCF calculation can also update the
original calculator's potential when it uses that calculator's
default physical state. If a run overrides its atoms or potential,
do not assume those one-time changes have become calculator defaults.
Use the result's potential or ``scf.calculator`` to choose the
state explicitly.

For the full first-calculation walkthrough, see :doc:`quickstart`.
For more examples of calculations beyond SCF and DOS, see the
`ASE2SPRKKR paper (Eddhib et al., 2026)
<https://arxiv.org/abs/2608.05957>`_ and :doc:`examples`.

To generate native input files **without executing SPR-KKR**:

.. code-block:: python

   from ase.build import bulk
   from ase2sprkkr import SPRKKR

   SPRKKR(atoms=bulk("Al", "fcc", a=4.05)).save_input(
       input_file="example.inp", potential_file="example.pot"
   )

``save_input()`` returns the prepared input parameters and the paths
of the input, potential and output files. The destination directory
must be persistent; ``directory=False`` is only for actual runs
using ``create_run()`` or ``calculate()``.

Filename templates
------------------

The calculator's ``input_file``, ``potential_file`` and
``output_file`` names can contain placeholders that are resolved
while preparing a calculation:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Placeholder
     - Replacement
   * - ``%a``
     - Chemical symbols / structure label derived from the atoms
   * - ``%t``, ``%T``
     - Task name, as supplied and in uppercase respectively
   * - ``%l``
     - Calculator label (empty if none was supplied)
   * - ``%d``
     - Local date and time (format ``YYYY-MM-DD_HH:MM``)
   * - ``%c``
     - Incrementing calculation counter
   * - ``%%``
     - Literal percent sign (for example, ``Fe%%T`` becomes ``Fe%T``)
   * - ``%_``
     - Underscore separator: omitted at the start of a filename;
       consecutive ``%_`` separators are collapsed

For example:

.. code-block:: python

   calculator = SPRKKR(
       atoms=bulk("Al", "fcc", a=4.05),
       label="test",
       input_file="%l%_%a_%T.inp",
       potential_file="%a.pot",
   )
   _, input_name, potential_name, _ = calculator.save_input()
   print(input_name, potential_name)

The values are resolved consistently for each prepared calculation,
including when multiple filenames use the same placeholder.
The ``%d`` format contains a colon, which is not portable in
Windows filenames; choose another template there.

``CONTROL.DATASET`` is a literal native filename prefix and may include
relative or absolute directories. ``CONTROL.ADSI`` is appended after an
underscore when nonempty; an empty ADSI adds no suffix. Native input
delimiters, whitespace, quoting, shell substitutions and wildcards cannot
be used in either identifier. DATASET cannot start with ``-`` or ``~``:
SPR-KKR passes derived TAU directory names to unquoted shell commands.
Hyphens inside paths, such as ``runs/-Fe``, are supported, as are literal
punctuation such as ``+@%!:^`` and UTF-8 names. Parent directories must
exist when SPR-KKR opens its output files.

Inspecting SCF results
----------------------

The SCF result contains parsed iterations, convergence information
and access to the newly written potential. After a calculation:

.. code-block:: python

   scf = SPRKKR(atoms=bulk("Al", "fcc", a=4.05)).calculate()
   print("Iterations:", len(scf.iterations))
   print("Converged:", scf.converged)
   print("Final convergence error:", scf.last_iteration.error())
   print("Final spin moment:", scf.last_iteration.moment.spin())
   print("Total energies:", scf.iterations.values_of(("energy", "ETOT")))

   print("Output potential:", scf.potential_filename)
   print("Charge data:", scf.potential.CHARGE())

The iteration data can be used to monitor convergence or to
postprocess many calculations. The final error and magnetic
moments come from the native output; whether the result is physically
converged must be checked separately from successful process
completion. Plotting the error history (with the optional ``plot``
dependencies installed) is also possible through
``scf.plot("error")``.

Results and further workflows
-----------------------------

``calculate()`` returns a task-specific ``TaskResult``, including
parsed quantities and paths to output files. You can also read
recognized result files independently:

.. code-block:: python

   from ase2sprkkr import ResultFile

   dos_data = ResultFile.from_file("calculation.dos")

See :doc:`result-files` for plotting, Grace files and NeXus export,
:doc:`remote` for MPI and execution profiles, and :doc:`execution`
for persistent calculations and reconnection. Command-line commands
are collected in :doc:`cli`, with runnable examples in :doc:`examples`.
