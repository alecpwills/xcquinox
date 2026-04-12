"""xcquinox.alec.config — architecture, feature, and pipeline spec dataclasses.

Implements THE SPEC §11.1: FeatureSpec, _FrozenDict, _FrozenTuple, _freeze,
and the FeatureSpec.as_kwargs thaw round-trip.
"""
from dataclasses import dataclass, field


def _freeze(value):
    """
    Recursively convert a value to a hashable form. Used by FeatureSpec to
    ensure kwarg dicts become fully hashable trees even if a caller passes
    a list or nested dict as a kwarg value.

    All three of dict, tuple, and list get distinct tagged subclasses
    (_FrozenDict, _FrozenTuple, plain unwrapped tuple-from-list) so the
    inverse _thaw is fully lossless: a dict round-trips to a dict, a
    tuple round-trips to a tuple, and a list round-trips to a list.
    """
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, dict):
        return _FrozenDict(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, tuple):
        return _FrozenTuple(_freeze(v) for v in value)
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    raise TypeError(
        f"FeatureSpec kwarg value of type {type(value).__name__} is not freezable: {value!r}"
    )


class _FrozenDict(tuple):
    """
    Tuple subclass used to tag a frozen dict so _thaw can losslessly
    distinguish it from a list-of-pairs. Behaves exactly like a tuple for
    hashing/equality/iteration purposes — only the type tag matters.
    """
    __slots__ = ()


class _FrozenTuple(tuple):
    """
    Tuple subclass used to tag an originally-tuple value so _thaw can
    distinguish it from an originally-list value. Plain tuple (no tag)
    means "was a list"; _FrozenTuple means "was a tuple". Behaves
    identically to tuple for hashing/equality/iteration.
    """
    __slots__ = ()


@dataclass(frozen=True)
class FeatureSpec:
    """Name + kwargs for materializing a Descriptor or Constraint by registry lookup."""
    name: str
    kwargs: _FrozenDict = field(default_factory=lambda: _FrozenDict())

    @classmethod
    def of(cls, spec):
        """
        Coerce (str | FeatureSpec | (str, dict)) -> FeatureSpec.

        Kwarg values are recursively frozen via _freeze so the resulting
        FeatureSpec is always hashable (lists become tuples, nested dicts
        become _FrozenDict tuples of (str, frozen-value) pairs).

        Raises:
            TypeError: if spec is not one of the supported forms.
        """
        if isinstance(spec, FeatureSpec):
            return spec
        if isinstance(spec, str):
            return cls(name=spec)
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
            name, kwargs_dict = spec
            if not isinstance(kwargs_dict, dict):
                raise TypeError(
                    f"FeatureSpec tuple form requires a dict of kwargs, got {type(kwargs_dict).__name__}"
                )
            frozen = _FrozenDict(sorted((str(k), _freeze(v)) for k, v in kwargs_dict.items()))
            return cls(name=name, kwargs=frozen)
        raise TypeError(f"cannot coerce {spec!r} to FeatureSpec")

    def as_kwargs(self) -> dict:
        """Thaw the frozen kwargs back into a dict for make_descriptor/make_constraint."""
        def _thaw(v):
            if isinstance(v, _FrozenDict):
                return {k: _thaw(val) for k, val in v}
            if isinstance(v, _FrozenTuple):
                return tuple(_thaw(e) for e in v)
            if isinstance(v, tuple):
                return [_thaw(e) for e in v]
            if isinstance(v, frozenset):
                return {_thaw(e) for e in v}
            return v
        return {k: _thaw(v) for k, v in self.kwargs}
