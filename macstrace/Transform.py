#%%
import xarray as xr
import numpy as np
import numpy.testing as npt
from numpy import deg2rad, rad2deg
from datetime import datetime
import mounttree.mounttree as mnt
import mounttree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import unittest as ut
from collections import defaultdict
#%%
class Sensor(object):
    def __init__(self, alt, act, times, name):
        self.alt=alt
        self.act=act
        self.times=times
        self.name=name
    
    @classmethod
    def from_file(cls, sensorfile):
        data=xr.open_dataset(sensorfile)
        return cls.from_dataset(data)
    @classmethod
    def from_dataset(cls, data):
        return cls(data.alt, data.act, data.time, data.source_name)

class Halo(object):
    def __init__(self, coord_universe,orientation, sensor1, sensor2):
        self.coord_universe=coord_universe
        self.orientation=orientation.sortby('time')
        self.sensors={'sensor1':sensor1, 'sensor2':sensor2}
        self.transform_cache={'sensor1':{}, 'sensor2':{}}
        self.reference_available=False
    
    def _add_reference_frame(self, time):
        frame=mnt.CartesianCoordinateFrame()
        frame.pos=[float(self.orientation['lat'].sel(time=time, method='nearest')),float(self.orientation['lon'].sel(time=time, method='nearest')),0]
        frame.euler=[0,0,0]
        frame.name='intersection_reference'
        self.coord_universe.get_frame('EARTH').add_child(frame)
        self.reference_available=True

    @classmethod
    def from_files(cls, mounttreefile, bahamasfile, sensor1file, sensor2file):
        geometry=xr.open_dataset(bahamasfile)
        coord_universe=mounttree.load_mounttree(mounttreefile)
        return cls(coord_universe, geometry, Sensor.from_file(sensor1file), Sensor.from_file(sensor2file))
    
    @classmethod
    def from_datasets(cls, mounttreefile, bahamasfile, data1, data2):
        geometry=xr.open_dataset(bahamasfile)
        coord_universe=mounttree.load_mounttree(mounttreefile)
        return cls(coord_universe, geometry, Sensor.from_dataset(data1), Sensor.from_dataset(data2))

    def _create_sensor_to_reference_transform(self,sensor, time):
        if not self.reference_available:
            self._add_reference_frame(time)
        keys=['lat', 'lon', 'height', 'yaw', 'pitch', 'roll']
        current_orientation={key:self.orientation[key].interp(time=time).values for key in keys}
        transformation=self.coord_universe.get_transformation(self.sensors[sensor].name.upper(), 'intersection_reference', **current_orientation)
        return transformation
    
    def get_transform(self, sensor, time):
        try:
            trans=self.transform_cache[sensor][time]
        except KeyError:
            trans=self._create_sensor_to_reference_transform(sensor, time)
            self.transform_cache[sensor][time]=trans
        return trans
    
    def altact_to_sensor(self,alt, act):
        """Convert along and across track angle to dir. vectors. Input in degrees!"""
        return [np.sin(deg2rad(act)), np.sin(deg2rad(alt))*np.cos(deg2rad(act)), np.cos(deg2rad(alt))*np.cos(deg2rad(act))]
    
    def dir_to_vza(self, vz):
        return np.rad2deg(np.arccos(vz))
    def dir_to_vaa(self, vx, vy):
        """Convert viewing direction vector (in NED) to viewing azimuth angle (0deg north, 90deg east)"""
        return np.rad2deg(np.arctan2(vy, vx))%360

    def _get_abs_view_direction_single(self, sensor, time):
        transform=self.get_transform(sensor, time)
        get_direction=lambda alt, act:np.stack(transform.apply_direction(*self.altact_to_sensor(alt,act)), axis=-1)
        direction=xr.apply_ufunc(get_direction, self.sensors[sensor].alt,self.sensors[sensor].act,output_core_dims=[['coords']])
        direction.coords['coords']=('coords', ['x', 'y', 'z'])
        direction=direction.expand_dims({'time':[time]})
        return direction

    def _get_abs_view_pos_single(self,sensor, time):
        transform=self.get_transform(sensor, time)
        position=xr.DataArray(np.stack(transform.apply_point(0,0,0), axis=-1),coords=[('coords', ['x', 'y', 'z'])])
        position=position.expand_dims({'time':[time]})
        return position

    def get_abs_view_direction(self, sensor):
        direction=xr.concat([self._get_abs_view_direction_single(sensor, time) for time in self.sensors[sensor].times.values], dim='time')
        return direction

    def get_abs_view_position(self, sensor):
        position=xr.concat([self._get_abs_view_pos_single(sensor, time) for time in self.sensors[sensor].times.values], dim='time')
        return position

    def get_abs_view_angles(self, sensor):
        direction=self.get_abs_view_direction(sensor)
        vaa=self.dir_to_vaa(direction.sel(coords='x'), direction.sel(coords='y'))
        vaa.name='vaa'
        vaa.attrs['units']='degree'
        vaa.attrs['long_name']='viewing azimuth angle'
        vaa.attrs['description']='North: 0°, East: 90°'
        vza=self.dir_to_vza(direction.sel(coords='z'))
        vza=vza.drop_vars('coords')
        vza.name='vza'
        vza.attrs['units']='degree'
        vza.attrs['long_name']='viewing zenith angle'
        vza.attrs['description']='Down: 0°, Up: 180°'
        return xr.Dataset({'vaa':vaa, 'vza':vza})
        
class Nearest_point(object):
    def __init__(self, *xr_features):
        assert(np.all([f.dims==xr_features[0].dims for f in xr_features]))
        assert(np.all([f.shape==xr_features[0].shape for f in xr_features]))
        self.fitcoords=xr_features[0].coords
        self.fitshape=xr_features[0].shape
        self.fitdims=xr_features[0].dims
        print("Building up kdtree...")
        self.nbs=NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self._flatten_combinexy(*tuple(map(lambda a:a.values, xr_features))))
    
    def _flatten_combinexy(self,*features):
        return np.column_stack(tuple(map(lambda a:a.flatten(), features)))

    def _predict_array(self,*features):
        assert(np.all([f.shape==features[0].shape for f in features]))
        n_ind=self.nbs.kneighbors(self._flatten_combinexy(*features), return_distance=False).reshape(features[0].shape)
        return n_ind
    
    def _unravel_fitindex(self, index):
        return np.stack(np.unravel_index(index, shape=self.fitshape), axis=-1)


    def nearest_index(self,*xr_features):
        print("Evaluating using kdtree...")
        index=xr.apply_ufunc(self._predict_array,*xr_features)
        index=xr.apply_ufunc(self._unravel_fitindex, index, output_core_dims=[['fitdim']])
        index.coords['fitdim']=('fitdim', list(self.fitdims))
        return index


class Transformer(object):
    def __init__(self,halo, shape):
        self.halo=halo
        self.shape=shape

    @classmethod
    def from_files(cls,mounttreefile, orientationfile, sensor1file, sensor2file, shape):
        halo=Halo.from_files(mounttreefile, orientationfile, sensor1file, sensor2file)
        return cls(halo, shape)
        
    @classmethod
    def from_datasets(cls, mounttreefile, orientationfile, data1, data2, shape):
        halo=Halo.from_datasets(mounttreefile, orientationfile, data1, data2)
        return cls(halo, shape)

    def get_points_on_cloud(self, sensor):
        print(f"Calculating directions for {sensor}...")
        directions=self.halo.get_abs_view_direction(sensor)
        print(f"Calculating positions for {sensor}...")
        positions=self.halo.get_abs_view_position(sensor)
        print(f"Calculating intersections for {sensor}...")
        intersections=self.shape.intersect(positions.sel(coords='x'),positions.sel(coords='y'),positions.sel(coords='z'),directions.sel(coords='x'),directions.sel(coords='y'),directions.sel(coords='z'))
        return intersections

    def _isel_coord_multiindex(self, da, **kwargs):
        result=da.transpose(*kwargs.keys()).values[tuple([a.values for a in kwargs.values()])]
        return xr.DataArray(result,coords=next(iter(kwargs.values())).coords)

    def transform(self, data):
        kdtree=Nearest_point(*self.get_points_on_cloud('sensor1'))
        index=kdtree.nearest_index(*self.get_points_on_cloud('sensor2'))
        result=self._isel_coord_multiindex(data, **{key:index.sel(fitdim=key) for key in data.dims})
        return result.drop_vars('fitdim')

