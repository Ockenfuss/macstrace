from macstrace.Transform import Halo, Transformer
from macstrace.Intersectors import Plane_intersector
import numpy.testing as npt
import unittest as ut
import numpy as np
import xarray as xr
class Test_CloudIntersector(ut.TestCase):
    def test_Plane_intersector(self):
        cloud=Plane_intersector(height=2000)
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

class TestTransform(ut.TestCase):
    def setUp(self):
        self.mountfile='test/fixtures/mounttree.yaml'
        self.geomfile='test/fixtures/geometry.nc'
        self.vnirfile='test/fixtures/vnir.nc'
        self.swirfile='test/fixtures/swir.nc'
        intersector=Plane_intersector(height=2000)
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