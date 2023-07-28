# GeoHeat-GB: A geospatial power systems planning model for heat electrification in Britain

GeoHeat-GB is an open-source power systems planning model for heat electrification in Britain with high spatial resolution. This model is built on the electricity-only [PyPSA-Eur](https://pypsa-eur.readthedocs.io) open model dataset of the European power system. 

GeoHeat-GB includes high spatial and temporal resolution electricity demand projections for residential heat pump adoption. The level of residential air- and ground-source heat pump adoption is exogenously determined and can be specified in the `config.yaml` file. This model also includes a high-resolution representation of the British power system and low-resolution resolution representation of interconnected grids. 

## Licenses and citation

Similarly to PyPSA, GeoHeat-GB is distributed under the MIT license.

When you use GeoHeat-GB, please cite the forthcoming paper:

GeoHeat-GB is based on the PyPSA-Eur open model dataset of the European power system. When using GeoHeat-GB, please also credit the authors of PyPSA-Eur following their [guidelines](https://pypsa-eur.readthedocs.io/en/latest/#citing-pypsa-eur).

## Setup

Clone the GeoHeat-GB repository using the following command in your terminal:
```
/some/other/path % cd /some/path

/some/path % git clone https://github.com/clairehalloran/GeoHeat-GB.git
```

Install the python dependencies (which are the same as those of PyPSA-Eur) using the package manager of your choice. When using `conda`, enter the following commands in your terminal to install and activate the environment:

```
.../GeoHeat-GB % conda env create -f envs/environment.yaml
.../GeoHeat-GB % conda activate pypsa-eur
```

Install a solver of your choice that is compatible with PyPSA following [these instructions](https://pypsa.readthedocs.io/en/latest/installation.html#getting-a-solver-for-optimisation).

The model can be configured in a similar way to PyPSA-Eur using the configuration file `config.yaml`. An example file with the heating options is included as `config.heat.yaml`. The configuration options added in the `heating` section are:

```
heating:
  cutout: europe-2019-era5
  single_GB_temperature: true
  heat_sources: [air, ground]
  heat_pump_sink_T: 55. # Celsius
  air:
    share: 0.75
  ground:
    share: 0.25
```
The heating `cutout` parameter provides name of the file used to create the [Atlite](https://atlite.readthedocs.io/en/latest/) cutout used to calulcate heating demand and COP values. 

The `single_GB_temperature` parameter provides the option to use spatially uniform temperatures to calculate heating demand and COP values in Britain. See forthcoming paper for detailed discussion. 

The `heat_pump_sink_T` parameter is the output temperature in degrees Celsius for all heat pumps considered and is used to calculate hourly COP values. 

For both air- and ground-source heat pumps, the share of British households using the technology can be specified with a value between 0 and 1 for the `share` parameter. A value of 0 indicates that no households use the technology, and a value of 1 indicates that all households use the technology. Currently technology adoption is uniform across all parts of Britain.

For additional configuration options, refer to the [PyPSA-Eur documentation on configuration](https://pypsa-eur.readthedocs.io/en/latest/configuration.html).

## Running the model

Like the PyPSA-Eur model, this model is built through a snakemake workflow. Users are referred to the [PyPSA-Eur documentation](https://pypsa-eur.readthedocs.io) for detailed instructions on running the model.

## Data
The model uses historical temperature data to project hourly residential heating at high spatial resolution using [heating demand profiles](https://figshare.com/articles/dataset/Monitored_heat_pump_heat_demand_profiles_-_supplementary_information_to_Watson_et_al_2021_/13547447) based on the Renewable Heat Premium Payment trials. The development of these profiles is described in the paper [How will heat pumps alter national half-hourly heat demands? Empirical modelling based on GB field trials](https://doi.org/10.1016/j.enbuild.2021.110777). These profiles are used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and have been modified from half-hourly to hourly to match the temporal resolution of other generation and demand data used in the model.

The model uses [high spatial resolution population data](https://catalogue.ceh.ac.uk/documents/0995e94d-6d42-40c1-8ed4-5090d82471e1) that contains data supplied by Natural Environment Research Council. ©NERC (Centre for Ecology & Hydrology). Contains National Statistics data © Crown copyright and database right 2011. These data are used under the [Open Government License](https://eidc.ceh.ac.uk/licences/open-government-licence-ceh-ons/plain). If you use this model, you must cite [UK gridded population 2011 based on Census 2011 and Land Cover Map 2015](https://doi.org/10.5285/0995e94d-6d42-40c1-8ed4-5090d82471e1).
