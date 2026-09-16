"""qlsgym -- quantum-logic-spectroscopy state-preparation gym."""
from .spec import (Action, BatchedEngine, Block, Molecule, Primitive, System,
                   TauBatchedEngine, Task, Trap, Window, SIGMAS, TWO_PI)

__all__ = ["Action", "BatchedEngine", "Block", "Molecule", "Primitive", "System",
           "TauBatchedEngine", "Task", "Trap", "Window", "SIGMAS", "TWO_PI",
           "load_molecule"]


def load_molecule(name: str, **kw) -> Molecule:
    """Registry entry point: "h3o", "thf", "synthetic"."""
    from .molecules import load
    return load(name, **kw)
