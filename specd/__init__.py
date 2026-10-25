"""SpecSWD APIs; optional backends are imported when requested."""

from importlib import import_module

_EXPORTS = {
    "SpecWorkSpace": (".specd", "SpecWorkSpace"),
    "THSolver": (".THSolver", "THSolver"),
    "constant_q_response": (".attenuation", "constant_q_response"),
    "constant_q_response_derivatives": (".attenuation", "constant_q_response_derivatives"),
    **{name: (".io", name) for name in (
        "ComplexCSR", "DispersionTable", "KernelTable", "RaggedComplexField", "SolverOutput")},
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in ("specd", "torch"):
        return import_module("." + name, __name__)
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
