![Logo](images/parastell-logo.svg)

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/svalinn/parastell?tab=MIT-1-ov-file#readme)

[![CI testing](https://github.com/svalinn/parastell/actions/workflows/ci.yml/badge.svg)](https://github.com/svalinn/parastell/actions/workflows/ci.yml)

[![Docker status](https://github.com/svalinn/parastell/actions/workflows/docker_publish.yml/badge.svg)](https://github.com/svalinn/parastell/actions/workflows/docker_publish.yml)

[![Build status](https://github.com/svalinn/parastell/actions/workflows/build.yml/badge.svg)](https://github.com/svalinn/parastell/actions/workflows/build.yml)

---

Open-source Python package featuring a parametric, 3-D, medium-fidelity modeling toolset for stellarator fusion devices. ParaStell has the following core capabilities:

- Model in-vessel components of uniform or non-uniform thickness using plasma equilibrium VMEC data or custom first wall data, and a user-defined radial build
- Model magnet coils using coil filament point-locus data and a user-defined cross-section
- Generate tetrahedral meshes of in-vessel components and magnets

ParaStell also offers the following neutronics support:

- Generate DAGMC geometries
- Generate tetrahedral neutron source definitions
- Calculate neutron wall-loading
- Generate energy-, direction-, particle-, and space-resolved OpenMC magnet-interface tallies and phase-space handoffs for local deterministic HTS models

![Example model](images/parastell-example.png)

## Dependencies
ParaStell depends on:

- [CadQuery](https://cadquery.readthedocs.io/en/latest/installation.html)
- [PyDAGMC](https://github.com/svalinn/pydagmc)
- [MOAB](https://bitbucket.org/fathomteam/moab/src/master/)
- [CAD-to-DAGMC](https://github.com/fusion-energy/cad_to_dagmc)
- [OpenMC](https://github.com/openmc-dev/openmc) 0.15.1 or later
- [h5py](https://www.h5py.org/)
- [NumPy](https://numpy.org/install/)
- [SciPy](https://scipy.org/install/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [Coreform Cubit](https://coreform.com/products/downloads/) (optional)

## Install ParaStell
Download and extract the ParaStell repository:

```bash
git clone git@github.com:svalinn/parastell.git
```

or download the ZIP file from the repository home page.

### Install Python Dependencies

This guide will use the conda package manager to install Python dependencies. Conda provides straight-forward installation of Python packages and switching between different collections of Python packages through the use of [environments](https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html).

If you have not already installed conda, you can use one of the following installers:
- [Miniforge](https://github.com/conda-forge/miniforge)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Anaconda](https://www.anaconda.com/)

A working conda environment with all ParaStell Python dependencies can be found in this repository's `environment.yml` file. To create the corresponding `parastell_env` conda environment, create the environment from the `environment.yml` file and activate the new environment:

```bash
conda env create -f environment.yml
conda activate parastell_env
```

### Install Coreform Cubit
To make use of ParaStell's Cubit functionality, download and install the latest version from [Coreform's Website](https://coreform.com/products/downloads/), then add the `/Coreform-Cubit-[version]/bin/` directory to your `PYTHONPATH` by adding a line similar to the following to your `.bashrc` file:

```bash
export PYTHONPATH=$PYTHONPATH:$HOME/Coreform-Cubit-[version]/bin/
```

Replace `$HOME` with the path to the Coreform Cubit directory. Additional information about adding modules to your `PYTHONPATH` can be found [here](https://www.tutorialspoint.com/How-to-set-python-environment-variable-PYTHONPATH-in-Linux).
While it is possible to use ParaStell with older versions of Cubit, additional steps not in this guide may be required.

If you do not have a Coreform Cubit license, you may be able to get one through [Cubit Learn](https://coreform.com/products/coreform-cubit/free-meshing-software/) at no cost.

### Finally Install Parastell

Now that all dependencies have been installed, you can install ParaStell with `pip`. Run the following command from the root of the ParaStell repository:

``` bash
pip install --no-deps .
```

## Executing ParaStell Scripts with YAML Input
While ParaStell can be imported as a module to make use of its Python API, ParaStell also has an executable to alternatively call functionality via command line. This executable uses a YAML configuration file as a command-line argument to define input parameters.

The executable can be run from command line with a corresponding YAML file argument. For example:

```bash
parastell config.yaml
```

See the executable's help message for more details.

## Reactor-to-magnet spectral handoff

ParaStell can augment an existing reactor-scale OpenMC/DAGMC model with stable magnet-interface tallies and optional surface-source banking. The outputs are intended as boundary inputs for a separate deterministic model that explicitly resolves the thin layers of an HTS winding pack.

Prepare a model without starting transport:

```bash
parastell-magnet-handoff prepare \
  --config examples/magnet_spectral_handoff.yaml \
  --model reactor_openmc_model \
  --output-dir reactor_magnet_run \
  --region winding_pack_01 \
  --direction both
```

Add `--run` for a local OpenMC execution, or run the prepared model through the normal HPC scheduler and post-process it afterward:

```bash
parastell-magnet-handoff postprocess \
  --config examples/magnet_spectral_handoff.yaml \
  --statepoint reactor_magnet_run/statepoint.100.h5 \
  --surface-source reactor_magnet_run/surface_source.h5 \
  --output-dir reactor_magnet_handoff \
  --region winding_pack_01 \
  --selection both
```

The exported products include normalized energy/direction-resolved tallies, local-coordinate phase-space records, uncertainty columns, and a versioned manifest. See [the full magnet spectral handoff guide](docs/magnet_spectral_handoff.md) for configuration, normalization, limitations, and validation requirements.

Run the branch's end-to-end planar, real-DAGMC-sector, and explicit HTS-layer validation gates with:

```bash
parastell-magnet-handoff-validate \
  --output-dir magnet_handoff_validation \
  --assets-dir tests/files_for_tests
```

The deterministic replay preserves the sampled position, direction, particle, and energy correlations while normalizing the bank to the companion OpenMC boundary-current tally. Its current implementation is an exact uncollided/removal thin-layer operator, not a substitute for a later scattering and PKA-response solver.

## Citing
If referencing ParaStell in a document or presentation, please cite the following publication:

- Connor A. Moreno, Aaron Bader, and Paul P.H. Wilson, "ParaStell: parametric modeling and neutronics support for stellarator fusion power plants," *Frontiers in Nuclear Engineering*, **3**:1384788 (2024). DOI: [10.3389/fnuen.2024.1384788](https://www.frontiersin.org/journals/nuclear-engineering/articles/10.3389/fnuen.2024.1384788/full)
