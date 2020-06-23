from macstrace.Transform import Cloud_intersector, Halo
import numpy.testing as npt
import unittest as ut
import numpy as np

class Test_CloudIntersector(ut.TestCase):
    def test_cloud_intersector(self):
        cloud=Cloud_intersector(height=2000)
        v=np.array([1,0,1])
        v=v/np.linalg.norm(v)
        npt.assert_almost_equal(cloud.intersect(10,0,-10000,*v),[8010,0])
        v=np.array([0,-1,1])
        v=v/np.linalg.norm(v)
        npt.assert_almost_equal(cloud.intersect(0,100,-10000,*v),[0,-7900])


class TestHalo(ut.TestCase):
    def test_reference_transform(self):
        halo=Halo.from_files('/project/meteo/work/Paul.Ockenfuss/Master/Specmacs/mounttrees/eurec4a/eurec4a_pretest_mounttree.yaml','/project/meteo/data/eurec4a/20190516/nas/adlr_20190516a-IGI-final-data-100Hz-SPECMAC_simplified.nc')
        halo._add_reference_frame(np.datetime64('2019-05-16T14:33:30'))
        time=np.datetime64('2019-05-16T14:33:50')
        transform_direction=halo.get_sensor_to_reference_transform(time)
        npt.assert_allclose(-1*halo.orientation['height'].sel(time=time, method='nearest'),transform_direction.apply_point(0,0,0)[2], atol=2)
