# GeoHeat-GB: A power systems planning model for heat electrification in Britain

GeoHeat-GB is an open-source power systems planning model for heat electrification in Britain. This model is built on the electricity-only [PyPSA-Eur]((https://pypsa-eur.readthedocs.io) open model dataset of the European power system. 

The model uses historical temperature data to project hourly residential heating at high spatial resolution using heating demand profiles based on the Renewable Heat Premium Payment trials. The development of these profiles is described in the paper [How will heat pumps alter national half-hourly heat demands? Empirical modelling based on GB field trials](https://doi.org/10.1016/j.enbuild.2021.110777).

As with the PyPSA-Eur model, this model is built through a snakemake workflow. Users are referred to the [PyPSA-Eur documentation](https://pypsa-eur.readthedocs.io) for instructions on running the model.
