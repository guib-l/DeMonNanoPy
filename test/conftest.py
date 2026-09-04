import numpy as np
import os,sys
import shutil
import pytest
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption(
        "--run-optional", action="store_true", default=False, help="run optional tests"
    )
    parser.addoption("--beta", action="store_true", default=False, help="run beta-features tests")
    parser.addoption(
        "--references", action="store_true", default=False, help="run references-features tests"
    )
    parser.addoption(
        "--forces", action="store_true", default=False, help="run forces-features tests"
    )


def pytest_collection_modifyitems(config, items):

    if not config.getoption("--run-optional"):
        skip_optional = pytest.mark.skip(reason="need --run-optional option")
        for item in items:
            if "optional" in item.keywords:
                item.add_marker(skip_optional)

    if not config.getoption("--beta"):
        skip_optional = pytest.mark.skip(reason="need --beta option")
        for item in items:
            if "beta" in item.keywords:
                item.add_marker(skip_optional)

    if not config.getoption("--references"):
        skip_optional = pytest.mark.skip(reason="need --references option")
        for item in items:
            if "references" in item.keywords:
                item.add_marker(skip_optional)

    if not config.getoption("--forces"):
        skip_optional = pytest.mark.skip(reason="need --forces option")
        for item in items:
            if "forces" in item.keywords:
                item.add_marker(skip_optional)


BOHR = 0.529177210544


def compute_numgrad(symbols, positions, calculator, delta=0.01):

    grad = np.zeros(np.shape(positions))

    tmp_positions = positions.copy()

    for i, atm in enumerate(positions):
        for j in range(3):
            tmp_positions[i, j] -= delta

            calculator.calculate(symbols=symbols, positions=tmp_positions)

            ea = calculator.results["energy"]["energy"]
            tmp_positions[i, j] += delta * 2

            calculator.calculate(symbols=symbols, positions=tmp_positions)
            eb = calculator.results["energy"]["energy"]

            tmp_positions[i, j] -= delta
            grad[i, j] = (0.5 * eb - 0.5 * ea) / (delta)

    return grad * BOHR


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        
        workdir = getattr(item.module, "WORKDIR", None)

        source = Path(workdir)
        destination = Path(os.path.join(workdir, f"failed_{item.name}"))

        destination.mkdir(exist_ok=True)

        for fichier in source.iterdir():
            if fichier.is_file():
                shutil.copy2(fichier, destination / fichier.name)
