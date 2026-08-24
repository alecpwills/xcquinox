"""xcquinox.alec.config: architecture, feature, and pipeline spec dataclasses.

Provides FeatureSpec, _FrozenDict, _FrozenTuple, _freeze, and the
FeatureSpec.as_kwargs thaw round-trip, together with MoleculeSpec,
PretrainSpec, TrainingSpec, and TestSpec.
"""
import os
from dataclasses import dataclass, field, fields


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
    hashing/equality/iteration purposes, only the type tag matters.
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
    plus kwargs dicts), not live instances. Descriptor and constraint instances
    are materialized only when a model is constructed."""
    name: str
    depth: int
    nodes: int
    attention: bool = False
    num_heads: int = 1
    descriptors: tuple[FeatureSpec, ...] = ()
    x_constraints: tuple[FeatureSpec, ...] = ()
    c_constraints: tuple[FeatureSpec, ...] = ()
    double_lob_clamp_allowed: bool = False
    # when True, the correlation network takes a spin-polarization input
    # feature (x1) and the model uses the zeta-dependent PW92 baseline (Dick &
    # Fernández-Serra 2021). Default False = unpolarized correlation (existing
    # behavior; checkpoints compatible). True is a NEW checkpoint family (cnet
    # input width +1) requiring retrain + re-pretrain.
    use_polarized_correlation: bool = False

    # Physics-correction flags (each is a NEW checkpoint family). Defaults
    # preserve byte-identical deserialization of old checkpoints; new registry
    # entries and cluster runs explicitly set True to opt in.
    #
    # dm_entropy_intensive: True divides the Shannon entropy by ln(max(n_occ, 2))
    # so dm_entropy is size-intensive (Dick & Fernández-Serra 2021 style). False
    # keeps the size-extensive ln(n_occ) form.
    dm_entropy_intensive: bool = False
    # descriptor_log_transform: True applies the Dick XCDiff
    # (1 - exp(-x²)) · log(x + 1) transform to s (XNet input) and rs+s (CNet
    # input), and to weighted_Z (CuspDescriptor feature 1). False feeds raw
    # values.
    descriptor_log_transform: bool = False
    # zero_init_final_layer: True zeros the final MLP layer's weight + bias at
    # construction so Fx = Fc = 1 exactly at init (the LDA/PW92 limit -- F=1
    # multiplies lda_x + PW92, NOT PBE). False keeps
    # Glorot init (gives Fx mean ~+2.65e-4 off 1).
    zero_init_final_layer: bool = False
    # meta_gga: DFS-faithful meta-GGA (PRB 104 L161109 Eq. 12-13). True switches the
    # X/C UEG gate to (x2 + tanh^2(x3)) (x3 = ln((alpha+1)/2)) and the exchange
    # Lieb-Oxford ceiling to 1.174; requires a "metagga" descriptor (which supplies
    # the iso-orbital alpha). A NEW checkpoint family. Default False -> unchanged.
    meta_gga: bool = False

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
        if not isinstance(self.use_polarized_correlation, bool):
            raise TypeError(
                f"ArchitectureConfig.use_polarized_correlation must be a plain "
                f"Python bool, got {type(self.use_polarized_correlation).__name__}"
            )
        for bool_field in ("dm_entropy_intensive", "descriptor_log_transform",
                            "zero_init_final_layer"):
            value = getattr(self, bool_field)
            if not isinstance(value, bool):
                raise TypeError(
                    f"ArchitectureConfig.{bool_field} must be a plain Python "
                    f"bool, got {type(value).__name__}"
                )
        if not isinstance(self.num_heads, int) or isinstance(self.num_heads, bool):
            raise TypeError(
                f"ArchitectureConfig.num_heads must be a plain Python int, "
                f"got {type(self.num_heads).__name__}"
            )
        if self.num_heads < 1:
            raise ValueError(
                f"ArchitectureConfig.num_heads must be >= 1, got {self.num_heads}"
            )
        if self.attention and self.nodes % self.num_heads != 0:
            raise ValueError(
                f"ArchitectureConfig: attention=True requires "
                f"nodes ({self.nodes}) divisible by num_heads ({self.num_heads})"
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
        # The rung is one fact, so its two statements must agree HERE, at
        # construction, rather than being reconciled by whichever reader gets
        # there first. Both directions are refused: the flag without the
        # descriptor has no alpha to gate on, and the descriptor without the
        # flag used to be resolved onto the SCAN parent density by
        # `resolve_parent_density` while `run_pretrain` fitted it to the PBE
        # targets from the flag alone -- 24.0 mHa per system off its parent,
        # reported by nothing.
        _descriptor_names = tuple(spec.name for spec in self.descriptors)
        if bool(self.meta_gga) != ArchitectureConfig.is_meta_gga(self):
            raise ValueError(
                f"ArchitectureConfig {self.name!r}: meta_gga="
                f"{bool(self.meta_gga)} disagrees with its descriptor list "
                f"{_descriptor_names!r}, which "
                + ("carries" if ArchitectureConfig.is_meta_gga(self)
                   else "does not carry")
                + " the 'metagga' descriptor. The meta-GGA rung is one fact: "
                "the flag switches the DFS UEG gate and the Lieb-Oxford "
                "ceiling, the descriptor supplies the iso-orbital alpha that "
                "gate reads, and the pretraining parent density, the "
                "enhancement-factor targets and the (s, alpha) mesh are all "
                "selected from it. Give the architecture both or neither."
            )

    @staticmethod
    def is_meta_gga(arch) -> bool:
        """Whether ``arch`` is on the meta-GGA rung: the ONE predicate.

        Defined as "the ``metagga`` descriptor is present", because that
        descriptor is what supplies the iso-orbital alpha the rung is made of;
        the ``meta_gga`` flag is the same fact stated twice, and
        :meth:`__post_init__` refuses an architecture on which the two
        disagree, so either reading answers the same for anything
        constructible.

        One definition because the question used to be asked two ways.
        ``pretrain_data_gen.resolve_parent_density`` read the flag OR the
        descriptor while ``pretrain.run_pretrain`` selected the
        enhancement-factor targets, the per-system parent-energy keys and the
        (s, alpha) mesh from the flag ALONE, so an architecture carrying the
        descriptor without the flag was fitted to PBE targets on the SCAN
        self-consistent density -- measured 24.0 mHa per system off its
        parent, with nothing in the run reporting a disagreement.

        A static method rather than a property because the callers include
        code that receives arch-LIKE objects (the energy-weight sweep's
        consistency check, test doubles), and the predicate has to answer for
        those too rather than turning into an AttributeError.
        """
        return any(getattr(spec, "name", None) == "metagga"
                   for spec in getattr(arch, "descriptors", ()))

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
        # +1 for the spin-polarization (x1) input on the correlation net.
        return 2 + (1 if self.use_polarized_correlation else 0) + self.n_extra_features

    def materialize_descriptors(self):
        from xcquinox.alec.descriptors import make_descriptor
        # Inject the arch-level flags into the descriptor kwargs
        # when the descriptor type recognizes them. ``intensive`` lives on
        # DMStatisticsDescriptor; ``log_transform`` on CuspDescriptor. Other
        # descriptors ignore the unknown kwargs (filtered by make_descriptor
        # via FeatureSpec.as_kwargs: known kwargs only).
        out = []
        for s in self.descriptors:
            kwargs = dict(s.as_kwargs())
            # dm_statistics no longer takes `intensive`: it only ever
            # normalized dm_entropy, which was removed 2026-08-06. The
            # ArchitectureConfig field is KEPT (frozen dataclass; the live
            # cluster array's spec files carry it and must still unpickle) but
            # is now inert.
            if s.name == "cusp" and "log_transform" not in kwargs:
                kwargs["log_transform"] = self.descriptor_log_transform
            out.append(make_descriptor(s.name, **kwargs))
        return tuple(out)

    def materialize_x_constraints(self):
        from xcquinox.alec.constraints import make_constraint
        return tuple(make_constraint(s.name, **s.as_kwargs()) for s in self.x_constraints)

    def materialize_c_constraints(self):
        from xcquinox.alec.constraints import make_constraint
        return tuple(make_constraint(s.name, **s.as_kwargs()) for s in self.c_constraints)

    @classmethod
    def from_spec(cls, name, depth, nodes, *, attention=False,
                  num_heads=None,
                  descriptors=(), x_constraints=(), c_constraints=(),
                  allow_scaling_symmetric_on_c: bool = False,
                  allow_double_lob_clamp: bool = False,
                  use_polarized_correlation: bool = False,
                  dm_entropy_intensive: bool = False,
                  descriptor_log_transform: bool = False,
                  meta_gga: bool = False,
                  zero_init_final_layer: bool = False):
        """Factory that accepts str | (str, dict) | FeatureSpec for each entry.

        ``num_heads`` is required when ``attention=True`` (no silent default,
        callers must specify the head count explicitly so per-architecture
        defaults are visible at the call site). When ``attention=False``, the
        value is ignored and stored as 1.
        """
        if attention and num_heads is None:
            raise ValueError(
                "ArchitectureConfig.from_spec: attention=True requires "
                "explicit num_heads (no silent default; specify the head "
                "count per architecture at the call site)."
            )
        resolved_heads = num_heads if num_heads is not None else 1
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
                    "allow_scaling_symmetric_on_c=True -- this destroys the "
                    "correlation network's rs dependence.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        descriptor_specs = tuple(FeatureSpec.of(x) for x in descriptors)
        if meta_gga and not any(s.name == "metagga" for s in descriptor_specs):
            raise ValueError(
                "ArchitectureConfig.from_spec: meta_gga=True requires a 'metagga' "
                "descriptor (it supplies the iso-orbital alpha the DFS gate reads)."
            )

        has_lob_constraint = any(s.name == "lieb_oxford" for s in x_spec_tuple)
        if has_lob_constraint and allow_double_lob_clamp:
            warnings.warn(
                "LiebOxfordBound is registered under x_constraints AND "
                "allow_double_lob_clamp=True -- the network will retain its "
                "built-in LOB wrap (lob_lim=1.804) in addition to the "
                "constraint, narrowing the effective F range from [0, 1.804] "
                "to [0.321, 1.613].",
                RuntimeWarning,
                stacklevel=2,
            )

        return cls(
            name=name, depth=depth, nodes=nodes, attention=attention,
            num_heads=resolved_heads,
            descriptors=descriptor_specs,
            x_constraints=x_spec_tuple,
            c_constraints=c_spec_tuple,
            double_lob_clamp_allowed=allow_double_lob_clamp,
            use_polarized_correlation=use_polarized_correlation,
            dm_entropy_intensive=dm_entropy_intensive,
            descriptor_log_transform=descriptor_log_transform,
            meta_gga=meta_gga,
            zero_init_final_layer=zero_init_final_layer,
        )


# ---------------------------------------------------------------------------
# ARCHITECTURES registry: the notebook-derived arch variants
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    "shallow":             ArchitectureConfig(name="shallow",      depth=2, nodes=8),
    "shallow_attn":        ArchitectureConfig(name="shallow_attn", depth=2, nodes=8,  attention=True, num_heads=2),
    "medium":              ArchitectureConfig(name="medium",       depth=3, nodes=16),
    "medium_attn":         ArchitectureConfig(name="medium_attn",  depth=3, nodes=16, attention=True, num_heads=4),
    # Each deep_* entry enables physics-correction flags
    # (dm_entropy_intensive, descriptor_log_transform, zero_init_final_layer).
    # Built via ArchitectureConfig.from_spec with True defaults; old pickled
    # specs without these fields unpickle to False for compatibility.
    "deep":                ArchitectureConfig.from_spec("deep",               4, 32,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_attn":           ArchitectureConfig.from_spec("deep_attn",          4, 32,
                              attention=True, num_heads=4,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_cusp":           ArchitectureConfig.from_spec("deep_cusp",          4, 32,
                              descriptors=["cusp"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_cusp_attn":      ArchitectureConfig.from_spec("deep_cusp_attn",     4, 32,
                              attention=True, num_heads=4,
                              descriptors=["cusp"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_dm":             ArchitectureConfig.from_spec("deep_dm",            4, 32,
                              descriptors=["dm_statistics"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_dm_attn":        ArchitectureConfig.from_spec("deep_dm_attn",       4, 32,
                              attention=True, num_heads=4,
                              descriptors=["dm_statistics"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_combined":       ArchitectureConfig.from_spec("deep_combined",      4, 32,
                              descriptors=["dm_statistics", "cusp"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_combined_attn":  ArchitectureConfig.from_spec("deep_combined_attn", 4, 32,
                              attention=True, num_heads=4,
                              descriptors=["dm_statistics", "cusp"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    # Notransform variants: no DM/Cusp descriptors and descriptor_log_transform
    # =False. Baseline comparison without Dick XCDiff features.
    # dm_entropy_intensive is a no-op here (no DM descriptor) but kept True for
    # consistency; zero_init_final_layer stays True (good init hygiene).
    "deep_notransform":       ArchitectureConfig.from_spec("deep_notransform",      4, 32,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=False,
                              zero_init_final_layer=True),
    "deep_notransform_attn":  ArchitectureConfig.from_spec("deep_notransform_attn", 4, 32,
                              attention=True, num_heads=4,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=False,
                              zero_init_final_layer=True),
    # 2026-06-20: depth-3/width-16 twins of the 8 dfs_step7 sweep archs. The
    # 2026-06-20 review found our 4x32 nets (~3.3k params) overfit
    # the tiny 26-point DFS pool; DFS used 3 hidden layers x 16 nodes (~0.6k).
    # Each twin mirrors its 4x32 sibling's flags EXACTLY, changing only capacity.
    "deep_3x16":                ArchitectureConfig.from_spec("deep_3x16",               3, 16,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_attn_3x16":           ArchitectureConfig.from_spec("deep_attn_3x16",          3, 16,
                              attention=True, num_heads=4,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_cusp_3x16":           ArchitectureConfig.from_spec("deep_cusp_3x16",          3, 16,
                              descriptors=["cusp"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_dm_3x16":             ArchitectureConfig.from_spec("deep_dm_3x16",            3, 16,
                              descriptors=["dm_statistics"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_combined_3x16":       ArchitectureConfig.from_spec("deep_combined_3x16",      3, 16,
                              descriptors=["dm_statistics", "cusp"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_combined_attn_3x16":  ArchitectureConfig.from_spec("deep_combined_attn_3x16", 3, 16,
                              attention=True, num_heads=4,
                              descriptors=["dm_statistics", "cusp"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    # Rung-3.5 localized-DM archs (ADDITIVE). The leaky deep_dm/deep_combined
    # entries above are KEPT so a pending in-flight array task still resolves
    # them. deep_rung35_3x16 (cusp + localized rung-3.5 DM occupancy) replaces
    # deep_combined in the sweep; deep_rung35only_3x16 (rung-3.5 alone) replaces
    # deep_dm. The rung-3.5 occupancy is leak-free + self-consistent (Janesko
    # arXiv:2206.07118; M11plus, Verma 2019). See descriptors.DMRung35Descriptor.
    "deep_rung35_3x16":         ArchitectureConfig.from_spec("deep_rung35_3x16",        3, 16,
                              descriptors=["cusp", "rung35"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_rung35_attn_3x16":    ArchitectureConfig.from_spec("deep_rung35_attn_3x16",   3, 16,
                              attention=True, num_heads=4,
                              descriptors=["cusp", "rung35"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    # Multi-width rung-3.5 (2026-08-06). ADDITIVE: the single-width entries
    # above are untouched, so an in-flight array task still resolves them. The
    # radial generalization of the localized DM projection; 3 widths x 2 spins
    # = 6 features.
    "deep_rung35ms_3x16":       ArchitectureConfig.from_spec("deep_rung35ms_3x16",     3, 16,
                              descriptors=["cusp", "rung35_multishell"],
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_rung35only_3x16":     ArchitectureConfig.from_spec("deep_rung35only_3x16",    3, 16,
                              descriptors=["rung35"],
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    # DFS-faithful meta-GGA archs (ADDITIVE). meta_gga=True switches the X/C UEG
    # gate to DFS's (x2 + tanh^2(x3)) prefactor (x3 = ln((alpha+1)/2)) and the
    # exchange Lieb-Oxford ceiling to 1.174; the "metagga" descriptor supplies the
    # iso-orbital alpha = (tau - tau_W)/tau_unif (DFS PRB 104 L161109 Eq. 6, 12-13).
    # These archs PRETRAIN TO SCAN (a GGA cannot fit SCAN's alpha-dependence).
    # deep_mgga_3x16 is the pure DFS meta-GGA (exchange on (s, alpha));
    # deep_rung35_mgga_3x16 stacks cusp + localized rung-3.5 DM + meta-GGA alpha to
    # test whether the extra richness helps (replaces deep_rung35only in the sweep).
    "deep_mgga_3x16":           ArchitectureConfig.from_spec("deep_mgga_3x16",          3, 16,
                              descriptors=["metagga"], meta_gga=True,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_mgga_attn_3x16":      ArchitectureConfig.from_spec("deep_mgga_attn_3x16",     3, 16,
                              attention=True, num_heads=4,
                              descriptors=["metagga"], meta_gga=True,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_rung35_mgga_3x16":    ArchitectureConfig.from_spec("deep_rung35_mgga_3x16",   3, 16,
                              descriptors=["cusp", "rung35", "metagga"], meta_gga=True,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    # The mgga stacking completions (2026-08-10, the third sweep arm):
    # cusp+metagga isolates "does cusp help the meta-GGA rung" (the
    # deep_cusp vs deep chain, lifted to rung 3); cusp+multishell+metagga is
    # the multi-width A/B against deep_rung35_mgga at the same rung. Both
    # pretrain to SCAN WITHOUT the (s, alpha) mesh -- a mesh node has no
    # geometry, so it cannot define cusp or projection columns (the same
    # documented caveat deep_rung35_mgga carries).
    "deep_cusp_mgga_3x16":      ArchitectureConfig.from_spec("deep_cusp_mgga_3x16",     3, 16,
                              descriptors=["cusp", "metagga"], meta_gga=True,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_rung35ms_mgga_3x16":  ArchitectureConfig.from_spec("deep_rung35ms_mgga_3x16", 3, 16,
                              descriptors=["cusp", "rung35_multishell", "metagga"],
                              meta_gga=True,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=True,
                              zero_init_final_layer=True),
    "deep_notransform_3x16":       ArchitectureConfig.from_spec("deep_notransform_3x16",      3, 16,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=False,
                              zero_init_final_layer=True),
    "deep_notransform_attn_3x16":  ArchitectureConfig.from_spec("deep_notransform_attn_3x16", 3, 16,
                              attention=True, num_heads=4,
                              dm_entropy_intensive=True,
                              descriptor_log_transform=False,
                              zero_init_final_layer=True),
}
# NOTE: spin-polarization-aware correlation is NOT a separate entry in
# this registry (which mirrors the notebook's arch variants). Build one
# via ``ArchitectureConfig.from_spec(..., use_polarized_correlation=True)`` (or
# set the flag on any arch). Wiring it into the step7 grid sweep + the notebook
# ARCHITECTURES cell is handled in the step7-config step so code and notebook
# stay in sync (a NEW checkpoint family: cnet input width +1, retrain required).


def get_architecture(name: str) -> ArchitectureConfig:
    return ARCHITECTURES[name]


def list_architectures() -> list[str]:
    return sorted(ARCHITECTURES.keys())


# ---------------------------------------------------------------------------
# MoleculeSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoleculeSpec:
    """Immutable molecule identity. Hashable (frozen dataclass, tuple-of-pairs
    composition) so it can serve as a jit-static arg or dict key.

    The optional ``external_data_path`` points at an ``.npz`` file containing
    reference data that ``precompute_fixed_density_data`` cannot compute from
    the PBE SCF alone. Supported keys: ``dm_target`` (same shape as
    ``dm_pbe``), ``rho_ref_grid`` (same shape as ``rho_grid``),
    ``ref_density_method`` (string), and ``E_ref_literature`` (scalar).
    Unknown keys are rejected. Each key is
    optional: a partial ``.npz`` is valid (e.g., only ``E_ref_literature``
    for atoms where density matching is skipped).

    The optional ``grid_level`` pins the pyscf DFT grid level used by
    ``precompute_fixed_density_data``. ``None`` (the default) leaves pyscf's
    default (currently level 3 with ``nwchem_prune``). Setting an explicit
    integer is required when external reference data was computed on a
    non-default grid, e.g., the step3b / step 4 experiment uses
    ``grid_level=1`` so that ``precompute`` rebuilds the same coarse grid
    the caller used when writing ``rho_ref_grid`` to the external ``.npz``.
    Values accepted: 0..9, matching ``pyscf.dft.gen_grid.Grids.level``.
    """
    name: str
    atom: str            # pyscf-format, e.g. "H 0 0 0; H 0 0 0.74"
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0
    # Sorted tuple of (symbol, count) pairs, e.g. (("H", 2),) for H2.
    # MUST be a tuple-of-tuples (NOT a dict) because MoleculeSpec is frozen
    # and the auto-generated __hash__ hashes every field.
    atom_composition: tuple[tuple[str, int], ...] = ()
    # Optional path to an .npz with reference data. None = no external data
    # (dm_target/rho_ref_grid/E_ref_literature stay None in MoleculeData).
    external_data_path: str | None = None
    # Optional pyscf DFT grid level. None = pyscf default (currently 3).
    # Must match the grid level used to generate any reference data pointed
    # at by external_data_path, otherwise the shape validator in
    # _load_external_data will reject the file.
    grid_level: int | None = None

    def __post_init__(self) -> None:
        if self.grid_level is not None:
            if not isinstance(self.grid_level, int) or isinstance(self.grid_level, bool):
                raise TypeError(
                    f"MoleculeSpec.grid_level must be int or None, got "
                    f"{type(self.grid_level).__name__} = {self.grid_level!r}"
                )
            if not (0 <= self.grid_level <= 9):
                raise ValueError(
                    f"MoleculeSpec.grid_level must be in [0, 9], got "
                    f"{self.grid_level}"
                )

    @classmethod
    def from_dict(cls, *, name: str, atom: str,
                  atom_composition: dict[str, int] | tuple,
                  basis: str = "sto-3g", charge: int = 0,
                  spin: int = 0,
                  external_data_path: str | None = None,
                  grid_level: int | None = None) -> "MoleculeSpec":
        """Convenience constructor that accepts atom_composition as a dict
        and canonicalizes it into the sorted tuple-of-pairs form."""
        if isinstance(atom_composition, dict):
            comp = tuple(sorted(atom_composition.items()))
        else:
            comp = tuple(sorted(tuple(atom_composition)))
        return cls(name=name, atom=atom, basis=basis, charge=charge,
                   spin=spin, atom_composition=comp,
                   external_data_path=external_data_path,
                   grid_level=grid_level)

    @property
    def composition_dict(self) -> dict[str, int]:
        """Dict view of atom_composition."""
        return dict(self.atom_composition)


# ---------------------------------------------------------------------------
# Spec describe helpers
# ---------------------------------------------------------------------------

def _json_coerce(value):
    """Recursively convert tuples to lists so json.dumps accepts the result."""
    if isinstance(value, tuple):
        return [_json_coerce(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_coerce(v) for k, v in value.items()}
    return value


def _describe_spec(spec) -> dict:
    """Render a frozen-dataclass spec as a json-serializable dict.

    - Nested ArchitectureConfig -> its .name str.
    - tuple[MoleculeSpec, ...] -> list of m.name strs.
    - Everything else -> _json_coerce'd primitive/list form.
    """
    out: dict = {}
    for f in fields(spec):
        value = getattr(spec, f.name)
        if isinstance(value, ArchitectureConfig):
            out[f.name] = value.name
        elif (
            isinstance(value, tuple)
            and value
            and all(isinstance(v, MoleculeSpec) for v in value)
        ):
            out[f.name] = [m.name for m in value]
        elif isinstance(value, tuple) and not value and f.name in ("molecules",):
            out[f.name] = []
        elif value is not None and hasattr(value, "describe"):
            out[f.name] = value.describe()
        else:
            out[f.name] = _json_coerce(value)
    return out


# ---------------------------------------------------------------------------
# PretrainSpec
# ---------------------------------------------------------------------------

#: Parent densities the pretraining targets may sit on. "auto" resolves to the
#: architecture's rung baseline (SCAN for the meta-GGA rung, PBE otherwise).
#: The harness parser states the same set as ``grid_config._PARENT_DENSITIES``
#: -- it cannot import this module, which pulls JAX and equinox, on the login
#: node -- and the two are pinned equal by the harness test suite, so a value
#: the parser admits and this spec refuses cannot ship.
PARENT_DENSITIES = ("pbe", "scan", "auto")

#: Largest accepted PRNG seed. ``jax.random.PRNGKey`` wraps modulo 2**32
#: instead of raising, so a seed outside the range silently ALIASES another
#: run's initialization (PRNGKey(-1) == PRNGKey(2**32 - 1), PRNGKey(2**32) ==
#: PRNGKey(0)) while the metadata records the number that was written;
#: ``create_network_pair`` keys cnet at seed + 1, so the top of the range is
#: excluded too. Mirrored by ``grid_config._MAX_SEED``.
MAX_SEED = 2 ** 32 - 2


@dataclass(frozen=True)
class PretrainSpec:
    """Pretraining config. Plain frozen dataclass, NOT an eqx.Module, so float
    fields are ordinary Python float leaves and participate in the auto-generated
    __eq__ / __hash__."""
    arch: ArchitectureConfig
    data_dir: str
    checkpoint_dir: str
    n_steps: int = 1000
    lr_start: float = 1e-2
    lr_end: float = 1e-5
    lr_decay_start: float = 0.2   # fraction of steps before decay
    grad_clip: float = 1.0
    seed: int = 42
    # Loss weighting scheme used by the pretraining loop. Validated at
    # construction so mistypes surface immediately rather than deep in training.
    loss_weighting: str = "unweighted"
    # --- Pretraining protocol (spec Sections 3.2, 6, 7) -------------------
    # Every default reproduces the pre-protocol run exactly, so an existing
    # spec is unchanged and the new behavior is opt-in per YAML.
    #
    # The parent functional whose SELF-CONSISTENT density the pretrain data
    # sits on. "auto" resolves to the architecture's rung baseline (SCAN for
    # the meta-GGA rung, PBE otherwise); "pbe" keeps every architecture on the
    # PBE-density file, which is what every file written before this change
    # is.
    parent_density: str = "pbe"
    # Weight of the per-system energy term, in inverse Hartree^2. The term is
    # mean_s (E_xc^NN_s - E_xc^parent_s)^2, so w_E = 1 makes a 1 mHa mean
    # energy error worth 1e-6, the order of the converged point-wise residual.
    # 0.0 = the point-wise objective alone, byte-identical to the prior loss.
    energy_term_weight: float = 0.0
    # Fraction of the MULTI-NUCLEUS systems withheld from the fit and scored
    # between optimizer steps. 0.0 = no split and no stop criterion.
    validation_fraction: float = 0.0
    # Seed of the held-out permutation. Separate from ``seed`` (the network
    # initialization) so every architecture in a sweep holds out the same
    # systems and their validation numbers are comparable.
    validation_seed: int = 0
    # Optimizer steps between validations.
    validate_every: int = 50
    # Validations without improvement before training stops. 0 = no early
    # stop; the best weights are still the ones kept.
    patience: int = 0

    def __post_init__(self) -> None:
        if self.loss_weighting not in ("unweighted", "integration"):
            raise ValueError(
                f"loss_weighting must be 'unweighted' or 'integration', "
                f"got {self.loss_weighting!r}"
            )
        if self.parent_density not in PARENT_DENSITIES:
            raise ValueError(
                f"parent_density must be one of "
                f"{', '.join(repr(v) for v in PARENT_DENSITIES)}, got "
                f"{self.parent_density!r}"
            )

    def validate(self) -> None:
        """Raise ValueError if spec is inconsistent."""
        import math
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")
        # energy_term_weight and validation_fraction join the finiteness sweep
        # for the reason the certificate tolerances do: NaN satisfies neither
        # sense of an ordinary bound (nan < 0 and nan >= 1.0 are both False),
        # so it would pass every check below and then make every comparison
        # against it False downstream. An infinite loss weight is refused on
        # the same grounds, since the objective it defines is not a measurable
        # quantity.
        for field_name in ("lr_start", "lr_end", "lr_decay_start", "grad_clip",
                           "energy_term_weight", "validation_fraction"):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")
        if not (0.0 <= self.lr_decay_start <= 1.0):
            raise ValueError(f"lr_decay_start must be in [0, 1], got {self.lr_decay_start}")
        if self.lr_start < self.lr_end:
            raise ValueError(f"lr_start ({self.lr_start}) must be >= lr_end ({self.lr_end})")
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        if self.energy_term_weight < 0:
            raise ValueError(
                f"energy_term_weight must be >= 0, got "
                f"{self.energy_term_weight}")
        if not (0.0 <= self.validation_fraction < 1.0):
            raise ValueError(
                f"validation_fraction must be in [0, 1), got "
                f"{self.validation_fraction}")
        if self.validate_every <= 0:
            raise ValueError(
                f"validate_every must be > 0, got {self.validate_every}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        # The held-out permutation's seed. Bounded for the reason MAX_SEED
        # states: a value outside the range is not refused downstream, it
        # ALIASES another split while the record names the number written.
        if not (0 <= self.validation_seed <= MAX_SEED):
            raise ValueError(
                f"validation_seed must be in [0, {MAX_SEED}], got "
                f"{self.validation_seed}")
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"data_dir does not exist: {self.data_dir}")
        if os.path.exists(self.checkpoint_dir) and not os.path.isdir(self.checkpoint_dir):
            raise ValueError(
                f"checkpoint_dir exists but is not a directory: {self.checkpoint_dir}"
            )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def describe(self) -> dict:
        return _describe_spec(self)


# ---------------------------------------------------------------------------
# TrainingSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingSpec:
    """Training config. Same D-H6 float-equality contract as PretrainSpec."""
    arch: ArchitectureConfig
    molecules: tuple[MoleculeSpec, ...]
    targets: tuple[tuple[str, float], ...]
    atom_energies: tuple[tuple[str, float], ...]
    loss_name: str
    loss_kwargs: tuple[tuple[str, object], ...] = ()
    n_steps: int = 200
    lr_start: float = 1e-3
    lr_end: float = 1e-5
    lr_decay_start: float = 0.0
    grad_clip: float = 1.0
    # Decoupled L2 weight decay (adamw); 0.0 = no decay (byte-identical to the
    # former adam). Positive values regularize the over-capacity nets. 2026-06-20.
    weight_decay: float = 0.0
    pretrain_checkpoint: str | None = None
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42
    solver_config: object | None = None
    loss_metric: str = "absolute"
    balancing: object | None = None
    # PBE-anchor regularization (step 6). Direct fields rather
    # than loss_kwargs because PBEAnchorSample contains jnp.ndarrays that
    # `_freeze` refuses. `object | None` matches the `solver_config` /
    # `balancing` precedent.
    pbe_anchor_weight: float = 0.0
    pbe_anchor_sample: object | None = None
    # When True (default), validate() requires every element symbol
    # appearing in any compound's composition to ALSO appear as a
    # single-atom training molecule (so `_atomic_reg` and the AE-anchor
    # bookkeeping have a place to point). With the mixed-pool
    # subset design, atom anchors are restricted to the Dick H/Li set
    # by construction; setting this False disables the enforcement so
    # specs can carry C/N/O/F/... compounds without forcing single-atom
    # MoleculeSpecs for those elements.
    require_atom_anchors: bool = True
    # Optimizer update scheme:
    #   "batched": one full-batch optimizer step per training step over
    #                    ALL species at once, with the configured `balancing`
    #                    strategy (GradNorm etc). The historical default.
    #   "per_molecule": DFS/dpyscf-style stochastic updates: each epoch shuffles
    #                    the per-target groups (one AE compound, one BH76
    #                    reaction, one IP pair, one atom anchor) and takes ONE
    #                    optimizer step per group, with FIXED channel weights
    #                    (`channel_weights`). `n_steps` is interpreted as the
    #                    number of EPOCHS in this mode (total updates =
    #                    n_steps * n_groups). Breaks the full-batch compromise
    #                    that pins multi-target atomization-energy fits.
    update_scheme: str = "batched"
    # Opt-in (default OFF): pad every molecule in a group up to one common shape so the
    # de-fused per-molecule kernels collapse to one compile per spin-type; results-
    # neutral (see xcquinox.alec.padding). Threaded from HyperParams by spec_builder.
    pad_group_to_common_shape: bool = False
    # Fixed per-channel weights for `update_scheme="per_molecule"` (ignored in
    # "batched" mode, where `balancing` controls weighting). Empty -> the
    # density-dominant dpyscf-style default in train._DEFAULT_CHANNEL_WEIGHTS.
    channel_weights: tuple[tuple[str, float], ...] = ()
    # Held-out VALIDATION slice (WS3, 2026-06-20) for in-loop early-stop +
    # validation-best model selection. ``validation_molecules`` are the val-slice
    # MoleculeSpecs (built by spec_builder via the SAME atoms_to_mol_spec path the
    # training molecules use, pointing at the density-only val_refs_dir);
    # ``validation_reactions_path`` is the JSON of val reaction dicts (written by
    # prepare_inputs to ``<run_dir>/validation/val_reactions.json``). BOTH default
    # to a no-op (empty tuple + None) so existing specs stay byte-identical and
    # the per-molecule loop only validates when both are populated AND
    # ``validate_every > 0``. ``object | None`` matches the solver_config /
    # balancing precedent (validation_molecules is a tuple of frozen MoleculeSpecs
    # and participates in __eq__/__hash__ like ``molecules``).
    validation_molecules: tuple[MoleculeSpec, ...] = ()
    validation_reactions_path: str | None = None
    # In-loop validation cadence + early-stop (threaded from HyperParams like
    # weight_decay). ALL default to a NO-OP so existing specs stay byte-identical:
    # ``validate_every == 0`` disables in-loop validation entirely; ``patience ==
    # 0`` disables early-stop. ``val_frac`` records the split fraction used to
    # build validation_molecules (provenance; the actual split happens upstream
    # in prepare_inputs/spec_builder via eval_holdout.split_held_out).
    val_frac: float = 0.2
    validate_every: int = 0
    patience: int = 0
    early_stop_min_delta: float = 0.0
    # Periodic-resume checkpoint cadence (WS5, 2026-06-20): every
    # ``checkpoint_every`` completed EPOCHS the per_molecule loop writes a
    # ``resume_*`` checkpoint set (model + opt_state + RNG + trackers) so a run
    # killed by walltime/maintenance can RESUME from its last periodic snapshot.
    # Defaults to a NO-OP (0 => no resume writes), so existing runs/tests stay
    # byte-identical. Only the per_molecule loop honors it.
    checkpoint_every: int = 0

    @property
    def channel_weights_dict(self) -> dict[str, float]:
        return dict(self.channel_weights)

    @property
    def targets_dict(self) -> dict[str, float]:
        return dict(self.targets)

    @property
    def atom_energies_dict(self) -> dict[str, float]:
        return dict(self.atom_energies)

    @property
    def loss_kwargs_dict(self) -> dict:
        return dict(self.loss_kwargs)

    @classmethod
    def from_dicts(
        cls,
        *,
        arch: "ArchitectureConfig",
        molecules: tuple["MoleculeSpec", ...],
        targets: dict[str, float],
        atom_energies: dict[str, float],
        loss_name: str,
        loss_kwargs: dict | None = None,
        **kwargs,
    ) -> "TrainingSpec":
        """Ergonomic dict-based constructor. Sorts keys for determinism."""
        return cls(
            arch=arch,
            molecules=molecules,
            targets=tuple(sorted(targets.items())),
            atom_energies=tuple(sorted(atom_energies.items())),
            loss_name=loss_name,
            loss_kwargs=tuple(sorted((loss_kwargs or {}).items())),
            **kwargs,
        )

    def validate(self) -> None:
        """Raise ValueError if spec is inconsistent."""
        # Deferred local import, xcquinox.alec.losses does not exist until
        # Task 4.1 ships.
        from xcquinox.alec.losses import LOSS_REGISTRY, list_losses
        if self.loss_name not in LOSS_REGISTRY:
            raise ValueError(
                f"unknown loss {self.loss_name!r}; known: {list_losses()}"
            )
        if not self.molecules:
            raise ValueError("molecules must be non-empty")
        targets_dict = self.targets_dict
        atom_energies_dict = self.atom_energies_dict
        missing_targets = [m.name for m in self.molecules if m.name not in targets_dict]
        if missing_targets:
            raise ValueError(f"targets missing for molecules: {missing_targets}")
        if not atom_energies_dict:
            raise ValueError("atom_energies must be non-empty")

        atom_mol_syms = {
            next(iter(dict(m.atom_composition)))
            for m in self.molecules
            if sum(dict(m.atom_composition).values()) == 1
        }
        referenced_syms: set[str] = set()
        for m in self.molecules:
            referenced_syms.update(dict(m.atom_composition).keys())
        missing_atoms = sorted(referenced_syms - atom_mol_syms)
        if missing_atoms and self.require_atom_anchors:
            raise ValueError(
                "Every atomic species referenced by a molecule's composition must "
                "also appear as a single-atom training molecule. "
                f"Missing single-atom molecules for: {missing_atoms}. "
                "Add single-atom MoleculeSpec entries (e.g., MoleculeSpec('H', 'H', ...)) "
                "for each missing symbol, OR construct the TrainingSpec with "
                "`require_atom_anchors=False` if your loss config does not need "
                "single-atom MoleculeSpecs for these elements (e.g., the 2026-05-07 "
                "mixed-pool subset design which restricts atom anchors to H/Li only)."
            )
        missing_atom_energies = sorted(atom_mol_syms - set(atom_energies_dict.keys()))
        if missing_atom_energies:
            raise ValueError(
                "atom_energies dict is missing entries for atomic training molecules: "
                f"{missing_atom_energies}"
            )
        # ALL elements referenced in any molecule's composition must
        # appear in atom_energies, regardless of require_atom_anchors.
        # _ae_from_atoms in losses.py reads atom_energies[Z] for every element
        # in every compound; a missing key causes a runtime KeyError or a
        # silently-wrong AE even when single-atom MoleculeSpecs are not required.
        missing_ae_for_referenced = sorted(
            referenced_syms - set(atom_energies_dict.keys())
        )
        if missing_ae_for_referenced:
            raise ValueError(
                "atom_energies is missing entries for elements referenced in "
                "molecule compositions: "
                f"{missing_ae_for_referenced}. "
                "Every element symbol appearing in any molecule's atom_composition "
                "must have a corresponding atom_energies entry so that "
                "atomization-energy losses can be computed correctly."
            )

        n_compounds = sum(
            1 for m in self.molecules if sum(dict(m.atom_composition).values()) > 1
        )
        if n_compounds == 0:
            raise ValueError(
                "TrainingSpec requires at least one compound molecule "
                "(atom_composition summing to > 1); got only atomic molecules."
            )

        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")
        if self.update_scheme not in ("batched", "per_molecule"):
            raise ValueError(
                f"update_scheme must be 'batched' or 'per_molecule', got "
                f"{self.update_scheme!r}"
            )
        import math
        for field_name in ("lr_start", "lr_end", "lr_decay_start", "grad_clip"):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")
        for m_name, t_value in targets_dict.items():
            # reject bool before isfinite, math.isfinite(True) is True,
            # so without this check True/False would silently coerce to 1.0/0.0.
            if isinstance(t_value, bool):
                raise ValueError(
                    f"targets[{m_name!r}] must be a float, got bool {t_value!r}"
                )
            if not math.isfinite(t_value):
                raise ValueError(f"targets[{m_name!r}] must be finite, got {t_value}")
        for sym, ae_value in atom_energies_dict.items():
            # same bool-rejection as targets.
            if isinstance(ae_value, bool):
                raise ValueError(
                    f"atom_energies[{sym!r}] must be a float, got bool {ae_value!r}"
                )
            if not math.isfinite(ae_value):
                raise ValueError(f"atom_energies[{sym!r}] must be finite, got {ae_value}")
        if not (0.0 <= self.lr_decay_start <= 1.0):
            raise ValueError(
                f"lr_decay_start must be in [0, 1], got {self.lr_decay_start}"
            )
        if self.lr_start < self.lr_end:
            raise ValueError(
                f"lr_start ({self.lr_start}) must be >= lr_end ({self.lr_end})"
            )
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        if self.pretrain_checkpoint is not None and not os.path.isdir(
            self.pretrain_checkpoint
        ):
            raise ValueError(
                f"pretrain_checkpoint directory not found: {self.pretrain_checkpoint}"
            )
        if os.path.exists(self.checkpoint_dir) and not os.path.isdir(self.checkpoint_dir):
            raise ValueError(
                f"checkpoint_dir exists but is not a directory: {self.checkpoint_dir}"
            )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.loss_metric not in ("absolute", "relative"):
            raise ValueError(
                f"loss_metric must be 'absolute' or 'relative', got {self.loss_metric!r}"
            )
        if self.balancing is not None:
            from xcquinox.alec.balancing import TwoPhaseConfig
            if isinstance(self.balancing, TwoPhaseConfig):
                if self.balancing.phase1_steps >= self.n_steps:
                    raise ValueError(
                        f"phase1_steps ({self.balancing.phase1_steps}) must be < "
                        f"n_steps ({self.n_steps})"
                    )
                if self.balancing.phase1_loss not in LOSS_REGISTRY:
                    raise ValueError(
                        f"phase1_loss {self.balancing.phase1_loss!r} not in "
                        f"LOSS_REGISTRY; known: {list_losses()}"
                    )
        if self.loss_kwargs:
            import inspect
            loss_cls = LOSS_REGISTRY[self.loss_name]
            sig = inspect.signature(loss_cls.__init__)
            allowed = set(sig.parameters.keys()) - {"self", "molecules"}
            provided_dict = dict(self.loss_kwargs)
            unknown = set(provided_dict.keys()) - allowed
            if unknown:
                raise ValueError(
                    f"loss_kwargs contains unknown keys for {self.loss_name!r}: "
                    f"{sorted(unknown)} (allowed: {sorted(allowed)})"
                )
            for k, v in provided_dict.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if not math.isfinite(v):
                        raise ValueError(
                            f"loss_kwargs[{k!r}] must be finite, got {v}"
                        )

    def describe(self) -> dict:
        return _describe_spec(self)


# ---------------------------------------------------------------------------
# TestSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestSpec:
    """Test/evaluation config. Same frozen-dataclass/hashable contract."""
    model_checkpoint: str
    arch: ArchitectureConfig
    molecules: tuple[MoleculeSpec, ...]
    metrics: tuple[str, ...] = ("total_energy",)
    metric_kwargs: tuple[tuple[str, tuple[tuple[str, object], ...]], ...] = ()
    atom_energies: tuple[tuple[str, float], ...] = ()
    output_dir: str = "test_results"
    save_per_molecule: bool = True
    save_aggregate: bool = True
    solver_config: object | None = None
    # PBE-anchor regularization (step 6). Direct fields rather
    # than loss_kwargs because PBEAnchorSample contains jnp.ndarrays that
    # `_freeze` refuses. `object | None` matches the `solver_config` /
    # `balancing` precedent.
    pbe_anchor_weight: float = 0.0
    pbe_anchor_sample: object | None = None

    @property
    def metric_kwargs_dict(self) -> dict[str, dict]:
        return {name: dict(kw) for name, kw in self.metric_kwargs}

    @property
    def atom_energies_dict(self) -> dict[str, float]:
        return dict(self.atom_energies)

    @classmethod
    def from_dicts(
        cls,
        *,
        model_checkpoint: str,
        arch: "ArchitectureConfig",
        molecules: tuple["MoleculeSpec", ...],
        metrics: tuple[str, ...] = ("total_energy",),
        metric_kwargs: dict[str, dict] | None = None,
        atom_energies: dict[str, float] | None = None,
        **kwargs,
    ) -> "TestSpec":
        """Ergonomic dict-based constructor. Sorts keys for determinism."""
        mk_tuple: tuple[tuple[str, tuple[tuple[str, object], ...]], ...] = tuple(
            (name, tuple(sorted(kw.items())))
            for name, kw in sorted((metric_kwargs or {}).items())
        )
        return cls(
            model_checkpoint=model_checkpoint,
            arch=arch,
            molecules=molecules,
            metrics=metrics,
            metric_kwargs=mk_tuple,
            atom_energies=tuple(sorted((atom_energies or {}).items())),
            **kwargs,
        )

    def validate(self) -> None:
        """Raise ValueError if any referenced metric name or checkpoint is invalid."""
        # Deferred local import, xcquinox.alec.evaluation does not exist
        # until Task 5.3 ships.
        from xcquinox.alec.evaluation import METRIC_REGISTRY, list_metrics
        if not os.path.isfile(self.model_checkpoint):
            raise ValueError(f"model_checkpoint not found: {self.model_checkpoint}")
        if not self.molecules:
            raise ValueError("molecules must be non-empty")
        if not self.metrics:
            raise ValueError("metrics must be non-empty")
        unknown = [m for m in self.metrics if m not in METRIC_REGISTRY]
        if unknown:
            raise ValueError(f"unknown metrics: {unknown}; known: {list_metrics()}")
        if "atomization_energy" in self.metrics and not self.atom_energies:
            raise ValueError("atomization_energy metric requires atom_energies")
        import math
        for sym, ae_value in self.atom_energies_dict.items():
            if not math.isfinite(ae_value):
                raise ValueError(f"atom_energies[{sym!r}] must be finite, got {ae_value}")
        import inspect
        for metric_name, kw in self.metric_kwargs_dict.items():
            if metric_name not in self.metrics:
                raise ValueError(
                    f"metric_kwargs[{metric_name!r}] is set but "
                    f"{metric_name!r} is not in self.metrics "
                    f"({list(self.metrics)}). The kwargs would be silently "
                    "ignored at run_test time, fix the typo or add the "
                    "metric to self.metrics."
                )
            if metric_name not in METRIC_REGISTRY:
                continue
            metric_cls = METRIC_REGISTRY[metric_name]
            sig = inspect.signature(metric_cls.__init__)
            allowed = set(sig.parameters.keys()) - {"self"}
            provided = set(kw.keys())
            unknown_keys = provided - allowed
            if unknown_keys:
                raise ValueError(
                    f"metric_kwargs[{metric_name!r}] contains unknown keys: "
                    f"{sorted(unknown_keys)} (allowed: {sorted(allowed)})"
                )
        if os.path.exists(self.output_dir) and not os.path.isdir(self.output_dir):
            raise ValueError(
                f"output_dir exists but is not a directory: {self.output_dir}"
            )
        os.makedirs(self.output_dir, exist_ok=True)

    def describe(self) -> dict:
        return _describe_spec(self)
