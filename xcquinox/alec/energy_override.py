"""Per-molecule energy-stack injection hook (loss-agnostic de-fusing).

The shared energy-stack builders :func:`losses._compute_energies` and
:func:`losses._compute_energy_trajectories` normally evaluate every molecule's
SCF energy inside one traced graph -- so a training step over an ``N``-molecule
group fuses ``N`` SCF forward+backward passes into a single XLA kernel whose LLVM
codegen scales with ``N``. For large bases/grids that kernel exhausts host RAM at
compile time.

The de-fused gradient utility (:mod:`xcquinox.alec.defused_grad`) instead
computes the per-molecule energies OUTSIDE that graph (one small compile per
molecule shape, reused) and *injects* the resulting stack here, so the loss's
channel assembly consumes it as an ordinary array. This module is the injection
seam: a request-scoped override that the two builders consult. It is deliberately
a leaf module (no package imports) so any loss can depend on it without a cycle.

The override is keyed by form -- ``"scalar"`` for :func:`_compute_energies`
(shape ``(N,)``) and ``"trajectory"`` for :func:`_compute_energy_trajectories`
(shape ``(N, T)``). The utility injects only the form the loss consumes; a loss
that asks for the other form while an override is active raises loudly rather
than silently re-fusing (which would both defeat the memory fix and mis-route the
gradient).
"""
import contextvars

_OVERRIDE: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "xcquinox_energy_stack_override", default=None
)


def energy_override_active() -> bool:
    """True while a de-fused gradient pass has injected an energy stack."""
    return _OVERRIDE.get() is not None


def get_energy_override(kind: str):
    """Return the injected energy stack for ``kind`` (``"scalar"`` or
    ``"trajectory"``), or ``None`` when no override is active (the ordinary,
    fused computation path).

    Raises ``RuntimeError`` when an override IS active but ``kind`` was not the
    injected form -- the de-fuse utility must precompute exactly the form the
    loss consumes.
    """
    override = _OVERRIDE.get()
    if override is None:
        return None
    if kind not in override:
        raise RuntimeError(
            f"energy-stack override active but form {kind!r} was not injected "
            f"(injected forms: {sorted(override)}). The de-fused gradient "
            "utility must inject the energy form the loss actually consumes."
        )
    return override[kind]


def set_energy_override(mapping: dict):
    """Install an override mapping (form -> stack) for the current context.

    Returns a token to pass to :func:`reset_energy_override`.
    """
    return _OVERRIDE.set(dict(mapping))


def reset_energy_override(token) -> None:
    """Remove the override installed by :func:`set_energy_override`."""
    _OVERRIDE.reset(token)
