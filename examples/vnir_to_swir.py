import macstrace
from macstrace.Intersectors import Plane_intersector
import xarray as xr
intersector=Plane_intersector(height=2000)
mountfile='test/fixtures/mounttree.yaml'
geomfile='test/fixtures/geometry.nc'
vnirfile='test/fixtures/vnir.nc'
swirfile='test/fixtures/swir.nc'
data_swir=xr.open_dataset(swirfile).sel(method='nearest')
data_vnir=xr.open_dataset(vnirfile).sel(method='nearest')
vnir_corrected=macstrace.smacs1_to_smacs2(mountfile, geomfile, data_vnir, data_swir, intersector)
print(vnir_corrected)
