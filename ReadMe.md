# Macstrace

Module to map data between different sensors based on position on the observed object.

When evaluating data from multiple sensors of SpecMacs, e.g. VNIR and SWIR, we have to take into account their different viewing angles. If we know the precise geometry of the observed object, e.g. the surface or a cloud, we can calculate the closest matching observation of sensor 1 for each observation of sensor 2 based on plane orientation and position. 

## Usage
See the examples in the `examples` directory.