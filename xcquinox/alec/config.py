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


@dataclass(frozen=True)
class ArchitectureConfig:
    """Stores descriptor and constraint specifications (registry-name strings
    plus kwargs dicts) — not live instances. Descriptor and constraint instances
    are materialized only when a model is constructed."""
    name: str
    depth: int
    nodes: int
    attention: bool = False
    descriptors: tuple[FeatureSpec, ...] = ()
    x_constraints: tuple[FeatureSpec, ...] = ()
    c_constraints: tuple[FeatureSpec, ...] = ()
    double_lob_clamp_allowed: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(
                f"ArchitectureConfig.name must be a plain Python str, "
                f"got {type(self.name).__name__}"
            )
        if not self.name:
            raise ValueError("ArchitectureConfig.name must be non-empty")
        if not isinstance(self.depth, int) or isinstance(self.depth, bool):
            raise TypeError(
                f"ArchitectureConfig.depth must be a plain Python int, "
                f"got {type(self.depth).__name__}"
            )
        if self.depth < 1:
            raise ValueError(
                f"ArchitectureConfig.depth must be >= 1, got {self.depth}"
            )
        if not isinstance(self.nodes, int) or isinstance(self.nodes, bool):
            raise TypeError(
                f"ArchitectureConfig.nodes must be a plain Python int, "
                f"got {type(self.nodes).__name__}"
            )
        if self.nodes < 1:
            raise ValueError(
                f"ArchitectureConfig.nodes must be >= 1, got {self.nodes}"
            )
        if not isinstance(self.attention, bool):
            raise TypeError(
                f"ArchitectureConfig.attention must be a plain Python bool, "
                f"got {type(self.attention).__name__}"
            )
        if not isinstance(self.double_lob_clamp_allowed, bool):
            raise TypeError(
                f"ArchitectureConfig.double_lob_clamp_allowed must be a plain "
                f"Python bool, got {type(self.double_lob_clamp_allowed).__name__}"
            )
        for field_name in ("descriptors", "x_constraints", "c_constraints"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                raise TypeError(
                    f"ArchitectureConfig.{field_name} must be a tuple of "
                    f"FeatureSpec, got {type(value).__name__}"
                )
            for i, entry in enumerate(value):
                if not isinstance(entry, FeatureSpec):
                    raise TypeError(
                        f"ArchitectureConfig.{field_name}[{i}] must be a "
                        f"FeatureSpec (use FeatureSpec.of or from_spec for "
                        f"str/tuple coercion), got {type(entry).__name__} = "
                        f"{entry!r}"
                    )

    @property
    def x_has_lieb_oxford_constraint(self) -> bool:
        return any(s.name == "lieb_oxford" for s in self.x_constraints)

    @property
    def resolved_xnet_lob_lim(self) -> float | None:
        if self.x_has_lieb_oxford_constraint and not self.double_lob_clamp_allowed:
            return None
        return 1.804

    @property
    def resolved_cnet_lob_lim(self) -> float | None:
        return 2.0

    @property
    def n_extra_features(self) -> int:
        from xcquinox.alec.descriptors import make_descriptor
        return sum(
            make_descriptor(spec.name, **spec.as_kwargs()).n_features
            for spec in self.descriptors
        )

    @property
    def n_input_features(self) -> int:
        return 2 + self.n_extra_features

    def materialize_descriptors(self):
        from xcquinox.alec.descriptors import make_descriptor
        return tuple(make_descriptor(s.name, **s.as_kwargs()) for s in self.descriptors)

    def materialize_x_constraints(self):
        from xcquinox.alec.constraints import make_constraint
        return tuple(make_constraint(s.name, **s.as_kwargs()) for s in self.x_constraints)

    def materialize_c_constraints(self):
        from xcquinox.alec.constraints import make_constraint
        return tuple(make_constraint(s.name, **s.as_kwargs()) for s in self.c_constraints)

    @classmethod
    def from_spec(cls, name, depth, nodes, *, attention=False,
                  descriptors=(), x_constraints=(), c_constraints=(),
                  allow_scaling_symmetric_on_c: bool = False,
                  allow_double_lob_clamp: bool = False):
        """Factory that accepts str | (str, dict) | FeatureSpec for each entry."""
        import warnings

        x_spec_tuple = tuple(FeatureSpec.of(x) for x in x_constraints)
        c_spec_tuple = tuple(FeatureSpec.of(x) for x in c_constraints)

        for s in c_spec_tuple:
            if s.name == "scaling_symmetric":
                if not allow_scaling_symmetric_on_c:
                    raise ValueError(
                        "ScalingSymmetric is registered under c_constraints, but "
                        "uniform coordinate scaling is NOT an exact symmetry of "
                        "the correlation functional (LDA correlation depends on "
                        "rs = (3/(4\u03c0 \u03c1))^(1/3)). Applying this constraint to the "
                        "correlation network destroys its density dependence. "
                        "If you really want this (e.g. research into approximate "
                        "c-side symmetry), pass allow_scaling_symmetric_on_c=True."
                    )
                warnings.warn(
                    "ScalingSymmetric is applied to AlecGGA_CNet via "
                    "allow_scaling_symmetric_on_c=True \u2014 this destroys the "
                    "correlation network's rs dependence. See \u00a74.3 of the spec.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        has_lob_constraint = any(s.name == "lieb_oxford" for s in x_spec_tuple)
        if has_lob_constraint and allow_double_lob_clamp:
            warnings.warn(
                "LiebOxfordBound is registered under x_constraints AND "
                "allow_double_lob_clamp=True \u2014 the network will retain its "
                "built-in LOB wrap (lob_lim=1.804) in addition to the "
                "constraint, narrowing the effective F range from [0, 1.804] "
                "to [0.321, 1.613]. See \u00a74.3 of the spec.",
                RuntimeWarning,
                stacklevel=2,
            )

        return cls(
            name=name, depth=depth, nodes=nodes, attention=attention,
            descriptors=tuple(FeatureSpec.of(x) for x in descriptors),
            x_constraints=x_spec_tuple,
            c_constraints=c_spec_tuple,
            double_lob_clamp_allowed=allow_double_lob_clamp,
        )


# ---------------------------------------------------------------------------
# §11.2: ARCHITECTURES dict — the 12 notebook variants
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    "shallow":             ArchitectureConfig(name="shallow",      depth=2, nodes=8),
    "shallow_attn":        ArchitectureConfig(name="shallow_attn", depth=2, nodes=8,  attention=True),
    "medium":              ArchitectureConfig(name="medium",       depth=3, nodes=16),
    "medium_attn":         ArchitectureConfig(name="medium_attn",  depth=3, nodes=16, attention=True),
    "deep":                ArchitectureConfig(name="deep",         depth=4, nodes=32),
    "deep_attn":           ArchitectureConfig(name="deep_attn",    depth=4, nodes=32, attention=True),
    "deep_cusp":           ArchitectureConfig.from_spec("deep_cusp",          4, 32, descriptors=["cusp"]),
    "deep_cusp_attn":      ArchitectureConfig.from_spec("deep_cusp_attn",     4, 32, attention=True, descriptors=["cusp"]),
    "deep_dm":             ArchitectureConfig.from_spec("deep_dm",            4, 32, descriptors=["dm_statistics"]),
    "deep_dm_attn":        ArchitectureConfig.from_spec("deep_dm_attn",       4, 32, attention=True, descriptors=["dm_statistics"]),
    "deep_combined":       ArchitectureConfig.from_spec("deep_combined",      4, 32, descriptors=["dm_statistics", "cusp"]),
    "deep_combined_attn":  ArchitectureConfig.from_spec("deep_combined_attn", 4, 32, attention=True, descriptors=["dm_statistics", "cusp"]),
}


def get_architecture(name: str) -> ArchitectureConfig:
    return ARCHITECTURES[name]


def list_architectures() -> list[str]:
    return sorted(ARCHITECTURES.keys())
