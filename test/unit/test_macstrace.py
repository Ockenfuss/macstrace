#%%
from macstrace.Transform import Halo, Transformer, Nearest_point
from macstrace.Shapes import Plane
import numpy.testing as npt
import unittest as ut
import numpy as np
import xarray as xr
#%%
class Test_CloudIntersector(ut.TestCase):
    def test_Plane(self):
        cloud=Plane(height=2000)
        v=np.array([1,0,1])
        v=v/np.linalg.norm(v)
        npt.assert_almost_equal(cloud.intersect(10,0,-10000,*v),[8010,0])
        v=np.array([0,-1,1])
        v=v/np.linalg.norm(v)
        npt.assert_almost_equal(cloud.intersect(0,100,-10000,*v),[0,-7900])


class TestHalo(ut.TestCase):
    def setUp(self):
        self.mountfile='test/fixtures/mounttree.yaml'
        self.geomfile='test/fixtures/geometry.nc'
        self.vnirfile='test/fixtures/vnir.nc'
        self.swirfile='test/fixtures/swir.nc'
        self.halo=Halo.from_files(self.mountfile, self.geomfile, self.vnirfile, self.swirfile)

    def test_dir_to_vza(self):
        angles=[0,30,90,120,180]
        vza=self.halo.dir_to_vza(np.cos(np.deg2rad(angles)))
        npt.assert_allclose(angles, vza)
    
    def test_dir_to_vaa(self):
        x=[1,0,-1,0]#NED:north, east, south, west
        y=[0,1,0,-1]
        vaa=[0,90,180,270]
        npt.assert_allclose(self.halo.dir_to_vaa(x,y),vaa)

    def test_reference_transform(self):
        """Manually add reference frame. It is located at the surface, to halo height and sensor z position should always be roughly the same (times -1 due to NED)"""
        halo=Halo.from_files(self.mountfile, self.geomfile, self.vnirfile, self.swirfile)
        halo._add_reference_frame(np.datetime64('2019-05-16T14:33:30'))
        time=np.datetime64('2019-05-16T14:33:50')
        transform=halo.get_transform('sensor1',time)
        npt.assert_allclose(-1*halo.orientation['height'].sel(time=time, method='nearest'),transform.apply_point(0,0,0)[2], atol=2)
    
    def test_create_reference_frame(self):
        """Let Halo create the reference frame at the position of the first transformation request"""
        halo=Halo.from_files(self.mountfile, self.geomfile, self.vnirfile, self.swirfile)
        time=np.datetime64('2019-05-16T14:33:50')
        transform=halo.get_transform('sensor1',time)
        point=transform.apply_point(0,0,0)
        npt.assert_allclose(-1*halo.orientation['height'].sel(time=time, method='nearest'),point[2], atol=2)
        npt.assert_allclose([0,0], point[:2], atol=10)#Specmacs should be within about 10m distance to Bahamas

    def test_transform_cache(self):
        halo=Halo.from_files(self.mountfile, self.geomfile, self.vnirfile, self.swirfile)
        time=np.datetime64('2019-05-16T14:33:50')
        trans1=halo.get_transform('sensor2', time)
        trans2=halo.get_transform('sensor2', time)
        assert(trans1 is trans2 is halo.transform_cache['sensor2'][time])

class TestNearest_Point(ut.TestCase):
    def test_nearest(self):
        x=xr.DataArray([[0,0.01],[1,1.01]], coords=[('x', [0,1]), ('y', [0,1])])
        y=xr.DataArray([[0,1],[0.01,1.01]], coords=[('x', [0,1]), ('y', [0,1])])
        evalx=xr.DataArray([-1,0,1,0,1], coords=[('a', [1,2,3,4,5])])
        evaly=xr.DataArray([-1,0,0,1,1], coords=[('a', [1,2,3,4,5])])
        kdtree=Nearest_point(x,y)
        ind=kdtree.nearest_index(evalx, evaly)
        self.assertCountEqual(ind.dims,('a','fitdim'))
        self.assertCountEqual(ind.fitdim.values,('x', 'y'))
        npt.assert_equal(ind.sel(fitdim='x').values, [0,0,1,0,1])
        npt.assert_equal(ind.sel(fitdim='y').values, [0,0,0,1,1])

class TestTransform(ut.TestCase):
    def setUp(self):
        self.mountfile='test/fixtures/mounttree.yaml'
        self.geomfile='test/fixtures/geometry.nc'
        self.vnirfile='test/fixtures/vnir.nc'
        self.swirfile='test/fixtures/swir.nc'
        intersector=Plane(height=2000)
        self.transform=Transformer.from_files(self.mountfile, self.geomfile, self.vnirfile, self.swirfile, intersector)
    def test_isel_multiindex(self):
        da=xr.DataArray(np.arange(16).reshape((4,4)), coords=[('a',[1,2,3,4]),('b',[1,2,3,4])])
        ai=xr.DataArray(np.array([[2,3],[1,2]]), coords=[('a', [5,6]), ('c',[7,8])])
        bi=xr.DataArray(np.array([[0,1],[2,3]]), coords=[('a', [5,6]), ('c',[7,8])])
        checkresult=xr.DataArray(np.array([[8,13],[6,11]]), coords=ai.coords)
        result=self.transform._isel_coord_multiindex(da, a=ai, b=bi)
        npt.assert_equal(checkresult.values, result.values)
        self.assertCountEqual(result.dims, checkresult.dims)
        npt.assert_equal(checkresult.a.values, result.a.values)
        npt.assert_equal(checkresult.c.values, result.c.values)