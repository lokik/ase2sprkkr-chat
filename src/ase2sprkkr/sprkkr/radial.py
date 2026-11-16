import numpy as np
import copy

from ..common.decorators import cached_property


def _equivalent_meshes(left, right):
    """Compare spherical and full-potential mesh parameters, not identity."""
    if not all(hasattr(mesh, "to_tuple") and hasattr(mesh, "fullpot_tuple")
               for mesh in (left, right)):
        return False
    if left.to_tuple() != right.to_tuple():
        return False
    full_left = left.fullpot_tuple()
    full_right = right.fullpot_tuple()
    if full_left is None or full_right is None:
        return full_left is full_right
    return (
        full_left[:3] == full_right[:3]
        and np.array_equal(full_left[3], full_right[3])
    )


class RadialValue:
    def __init__(self, value, mesh, fullpot=None):
        self._value = np.asarray(value)
        self._mesh = mesh
        self.fullpot = fullpot

    @cached_property
    def interpolator(self):
        def inter(x):
            return self._mesh.interpolator(x)

        ndim = self._value.ndim
        if ndim == 1:
            return inter(self._value)

        shape = self._value.shape[0]

        inters = [inter(self._value[i]) for i in range(shape)]

        def multiinter(x):
            if isinstance(x, np.ndarray):
                sx = (shape,) + x.shape
            else:
                sx = (shape,)
            out = np.empty(sx)
            for i in range(shape):
                out[i] = inters[i](x)
            return out

        return multiinter

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        if mesh is self._mesh:
            return
        if not self._mesh:
            self._mesh = mesh
            return
        if self.fullpot and _equivalent_meshes(self._mesh, mesh):
            # Equivalent meshes must not interpolate away numerical precision,
            # especially when copying a native non-spherical potential.
            self._mesh = mesh
            return
        if self.fullpot:
            raise ValueError(
                "Non-spherical FULLPOT data cannot be remeshed: "
                "the native coefficients must remain on their original mesh"
            )
        value = self.interpolate(mesh.coors)
        self._mesh = mesh
        self._value = value

    def copy(self, mesh=None):
        out = copy.copy(self)
        # The radial values must be private even if the target is a distinct
        # but numerically equivalent mesh. Otherwise a SiteType copy would
        # mutate the original AtomicType through the shared NumPy array.
        out._value = np.copy(self._value)
        out.__dict__.pop("interpolator", None)
        if mesh is not None and self._mesh is not mesh:
            out.mesh = mesh
        return out

    def for_mesh(self, mesh):
        if mesh is self.mesh:
            return self
        return self.copy(mesh)

    def interpolate(self, coors):
        return self.interpolator(coors)

    def __call__(self, at):
        return self.interpolate(at)

    @property
    def raw_value(self):
        return self._value

    @property
    def fullpot_data(self):
        """Native non-spherical FULLPOT block, preserved without reinterpretation.

        Unlike the spherical 2-row array, the non-spherical coefficients are
        stored as an opaque native payload. This permits lossless file
        round-trips without assuming an undocumented coefficient layout.
        """
        return self._fullpot_data

    @fullpot_data.setter
    def fullpot_data(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("FULLPOT radial data must be a native text block or None")
        self._fullpot_data = value or None

    @property
    def fullpot(self):
        """Alias for the native non-spherical radial contribution."""
        return self.fullpot_data

    @fullpot.setter
    def fullpot(self, value):
        self.fullpot_data = value


class RadialCharge(RadialValue):
    pass


class RadialPotential(RadialValue):
    @staticmethod
    def from_munchen(value, mesh, z):
        return RadialPotential(value, mesh, z)

    @staticmethod
    def from_julich(value, mesh, z):
        v = np.empty_like(value)
        out = RadialPotential(v, mesh, z)
        out.julich = value
        return out

    def __init__(self, value, mesh, z, fullpot=None):
        super().__init__(value, mesh, fullpot=fullpot)
        self.z = z

    @property
    def vt(self):
        return self._value[0]

    @vt.setter
    def vt(self, val):
        self._value[0, :] = val
        self._clear()

    @property
    def bt(self):
        return self._value[1]

    @bt.setter
    def bt(self, val):
        self._value[1, :] = val
        self._clear()

    @property
    def munchen(self):
        """Potential in Munchen format"""
        return self._value

    @munchen.setter
    def munchen(self, value):
        self._value = value
        self._clear()

    @property
    def julich(self):
        """Potential in Julich format"""
        return np.stack((self.up, self.down))

    @julich.setter
    def julich(self, value):
        value = np.asarray(value)
        self._value = np.stack(
            (
                0.5 * (value[0] + value[1]) - 2 * self.zr,
                0.5 * (value[0] - value[1]),
            )
        )
        self._clear()

    @property
    def up(self):
        return self._value[0] + self._value[1] + 2 * self.zr

    @property
    def down(self):
        return self._value[0] - self._value[1] + 2 * self.zr

    def _clear(self):
        """Clear cache on a value change"""
        if "interpolator" in self.__dict__:
            del self.__dict__["interpolator"]

    @property
    def zr(self):
        return self.z / self._mesh.coors
