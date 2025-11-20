## Badges
<!--
(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/SDCCA/dgl_abm) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/SDCCA/dgl_abm)](https://github.com/SDCCA/dgl_abm) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-dgl_abm-00a3e3.svg)](https://www.research-software.nl/software/dgl_abm) [![workflow pypi badge](https://img.shields.io/pypi/v/dgl_abm.svg?colorB=blue)](https://pypi.python.org/project/dgl_abm/) |
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>)|
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=SDCCA_dgl_abm&metric=alert_status)](https://sonarcloud.io/dashboard?id=SDCCA_dgl_abm) |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=SDCCA_dgl_abm&metric=coverage)](https://sonarcloud.io/dashboard?id=SDCCA_dgl_abm) || Documentation                      | [![Documentation Status](https://readthedocs.org/projects/dgl_abm/badge/?version=latest)](https://dgl_abm.readthedocs.io/en/latest/?badge=latest) || **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](https://github.com/SDCCA/dgl_abm/actions/workflows/build.yml/badge.svg)](https://github.com/SDCCA/dgl_abm/actions/workflows/build.yml) |
| Citation data consistency          | [![cffconvert](https://github.com/SDCCA/dgl_abm/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/SDCCA/dgl_abm/actions/workflows/cffconvert.yml) || SonarCloud                         | [![sonarcloud](https://github.com/SDCCA/dgl_abm/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/SDCCA/dgl_abm/actions/workflows/sonarcloud.yml) || Link checker              | [![link-check](https://github.com/SDCCA/dgl_abm/actions/workflows/link-check.yml/badge.svg)](https://github.com/SDCCA/dgl_abm/actions/workflows/link-check.yml) |
-->
[![work in progress](https://img.shields.io/badge/status-work%20in%20progress-yellow)]()
[![pre-release](https://img.shields.io/badge/status-pre--release-orange)]()
[![github license badge](https://img.shields.io/github/license/SDCCA/dgl_abm)](https://github.com/SDCCA/dgl_abm)
#### Main Branch 
[![Build (main)](https://github.com/SDCCA/dgl_abm/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/SDCCA/dgl_abm/actions/workflows/build.yml?query=branch%3Amain)
[![Coverage Status (main)](https://coveralls.io/repos/github/SDCCA/dgl_abm/badge.svg?branch=main)](https://coveralls.io/github/SDCCA/dgl_abm?branch=main)
#### Development Branch 
[![Build (development)](https://github.com/SDCCA/dgl_abm/actions/workflows/build.yml/badge.svg?branch=development)](https://github.com/SDCCA/dgl_abm/actions/workflows/build.yml?query=branch%3Adevelopment)
[![Coverage Status (development)](https://coveralls.io/repos/github/SDCCA/dgl_abm/badge.svg?branch=development)](https://coveralls.io/github/SDCCA/dgl_abm?branch=development)
## About DGL-ABM

DGL-ABM is an agent-based modeling (ABM) framework that leverages the Deep Graph Library ([DGL](https://www.dgl.ai/)) to facilitate the simulation of complex systems. Combining the power of graph neural networks with traditional ABM techniques enables researchers and practitioners to model the interactions of agents within a networked environment. Development of DGL-ABM continues with the goal of providing additional features and tutorials that will reduce the technical understanding required for its use while maintaining its flexibility for developers. An ecosystem of plugins is also planned to extend its functionality to common use cases.

## Installation

> ⚠️ **Note:** DGL-ABM is currently under development and is not release-ready yet.

- Requirements:
  - Python 3.11 (3.10 for macOS ARM64)
  - Conda ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution))
  - Git

<details>
<summary>Windows (Python 3.11)</summary>

```console
# Clone repository
git clone https://github.com/SDCCA/dgl_abm

# Create and activate environment
conda env create -f environment.yaml -n dgl_abm_cpu
conda activate dgl_abm_cpu

# Run demo (optional)
python -m demo/run_cpu_default.py
```
</details>

<details>
<summary>Linux (Python 3.11)</summary>

```console

# Create and activate environment
conda env create -f environment.yaml -n dgl_abm_cpu
conda activate dgl_abm_cpu

# Run demo (optional)
python -m demo/run_cpu_default.py
```
</details>

<details>
<summary>macOS (Intel, Python 3.11)</summary>

```console

# Create and activate environment
conda env create -f environment.yaml -n dgl_abm_cpu
conda activate dgl_abm_cpu

# Run demo (optional)
python -m demo/run_cpu_default.py
```
</details>

<details>
<summary>macOS (ARM64, Python 3.10)</summary>

```console

# Create and activate environment
conda env create -f environment_macos.yaml -n dgl_abm_mac
conda activate dgl_abm_mac

# Install PyTorch and DGL
chmod +x ./scripts/install_dgl.sh
./scripts/install_dgl.sh dgl_abm_mac

# Run demo (optional)
python -m demo/run_cpu_default.py
```
</details>

<!--
## Documentation

Include a link to your project's full documentation here.

## Contributing

If you want to contribute to the development of dgl_abm,
have a look at the [contribution guidelines](CONTRIBUTING.md).

-->

## Contributors and Acknowledgments
This model code was designed and developed with support from the Netherlands eScience Center by the Dutch Research Council (NWO) under contract 27020G08, titled “Computing societal dynamics of climate change adaptation in cities” through the contributions of Meiert Grootes, Pranav Chandramouli, Thijs van Lankveld, Sara Alidoost, and Victoria Garibay. Special acknowledgements to Debraj Roy and Tatiana Filatova, Co-Principal Investigators of the project, for their guidance.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

### Contact
Victoria Garibay, Ph.D. - [Contact Form](https://vmgaribay.github.io/portfolio/contact_form.html) | [GitHub Profile](https://github.com/vmgaribay)

### Additional Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
