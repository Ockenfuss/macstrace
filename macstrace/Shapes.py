



class Intersector(object):
    def intersect(self,px, py, pz,vx, vy,vz):
        """Calculate intersection point on the cloud surface from starting position px, py, pz and starting direction vx, vy, vz of a ray. Coordinates are relative to a frame named 'intersection_reference', which is added by the Halo object automatically if not existent. This function is expected to vectorize in numpy style."""
        raise(NotImplementedError)

class Plane_intersector(Intersector):
    def __init__(self, height=2000):
        self.h=-height
    def intersect(self,px, py, pz,vx, vy,vz):
        """Coordinates in NED with origin at the surface"""
        t=(self.h-pz)/vz
        return px+t*vx, py+t*vy
