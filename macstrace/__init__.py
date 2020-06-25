from .Transform import Transformer
import xarray as xr
import numpy as np





def smacs1_to_smacs2(mountfile, orientationfile, data1, data2,shape):
    """Map radiance values observed from one SpecMacs sensor (vnir, swir) to the positions observed by the other based on cloud geometry and plane orientation."""
    blocksize=np.timedelta64(10, 's')
    offset=np.timedelta64(1, 's')
    blockstart=np.arange(data2.time.min().values, data2.time.max().values, blocksize)
    frames=[]
    for wvl_1 in data1.wavelength.values:
        wvlframes=[]
        print(wvl_1)
        for wvl_2 in data2.wavelength.values:
            print(wvl_2)
            timeframes=[]
            for starttime in blockstart:
                print(starttime)
                block_swir=slice(starttime, starttime+blocksize)
                block_vnir=slice(starttime-offset, starttime+blocksize+offset)
                d1=data1.sel(time=block_vnir).sel(wavelength=wvl_1, method='nearest')
                d2=data2.sel(time=block_swir).sel(wavelength=wvl_2)
                transform=Transformer.from_datasets(mountfile, orientationfile, d1, d2, intersector)
                dat1_corrected=transform.transform(d1.radiance)
                timeframes.append(dat1_corrected)
            wvlframes.append(xr.concat(timeframes, dim='time'))
        dat1_corrected=xr.concat(wvlframes,dim='wavelength')
        dat1_corrected=dat1_corrected.rename({'wavelength':'wvl_2'})
        dat1_corrected=dat1_corrected.expand_dims({'wavelength':[wvl_1]})
        frames.append(dat1_corrected)
    result=xr.concat(frames, dim='wavelength')
    result.attrs=data1.radiance.attrs
    result.name=data1.radiance.name
    result.wvl_2.attrs['description']='Spectral channel on the second sensor to which this channel was traced.'
    return result
