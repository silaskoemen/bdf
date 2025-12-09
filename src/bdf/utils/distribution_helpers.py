import importlib
import pkgutil


def import_all_distributions() -> None:
    pkg = importlib.import_module("bdf.distributions")
    for module in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        # exclude 'kdl' for now
        if module.name.endswith("kdl"):
            continue
        importlib.import_module(module.name)
