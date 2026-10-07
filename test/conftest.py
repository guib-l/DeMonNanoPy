import os
import shutil
from pathlib import Path

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--optional", action="store_true", default=False, help="run optional tests")
    parser.addoption("--beta", action="store_true", default=False, help="run beta-features tests")
    parser.addoption(
        "--references", action="store_true", default=False, help="run references-features tests"
    )
    parser.addoption(
        "--forces", action="store_true", default=False, help="run forces-features tests"
    )
    parser.addoption("--optim", action="store_true", default=False, help="run optim-features tests")
    parser.addoption(
        "--dynamics", action="store_true", default=False, help="run dynamics-features tests"
    )
    parser.addoption("--freq", action="store_true", default=False, help="run freq-features tests")
    parser.addoption("--slow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--mc", action="store_true", default=False, help="run mc-features tests")
    parser.addoption("--3OB", action="store", default=None, help="3OB parameters required")
    parser.addoption("--all", action="store_true")

    parser.addoption(
        "--write-output",
        action="store_true",
        default=False,
        help="write all input/output files",
    )


@pytest.fixture
def parameters_3ob(request):
    chemin = request.config.getoption("--3OB")

    if chemin is None:
        pytest.skip("No 3OB parameters provided with --3OB")

    return chemin


def pytest_collection_modifyitems(config, items):

    if config.getoption("--all"):
        config.option.optional = True
        config.option.beta = True
        config.option.references = True
        config.option.forces = True
        config.option.dynamics = True
        config.option.optim = True
        config.option.mc = True
        config.option.freq = True

    if not config.getoption("--optional"):
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

    if not config.getoption("--optim"):
        skip_optional = pytest.mark.skip(reason="need --optim option")
        for item in items:
            if "optim" in item.keywords:
                item.add_marker(skip_optional)

    if not config.getoption("--dynamics"):
        skip_optional = pytest.mark.skip(reason="need --dynamics option")
        for item in items:
            if "dynamics" in item.keywords:
                item.add_marker(skip_optional)

    if not config.getoption("--mc"):
        skip_optional = pytest.mark.skip(reason="need --mc option")
        for item in items:
            if "mc" in item.keywords:
                item.add_marker(skip_optional)

    if not config.getoption("--freq"):
        skip_optional = pytest.mark.skip(reason="need --freq option")
        for item in items:
            if "freq" in item.keywords:
                item.add_marker(skip_optional)

    if not config.getoption("--slow"):
        skip_optional = pytest.mark.skip(reason="need --slow option")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_optional)


BOHR = 0.529177210544


def compute_numgrad(symbols, positions, calculator, delta=0.01, triger_str="energy"):

    grad = np.zeros(np.shape(positions))

    tmp_positions = positions.copy()

    for i, atm in enumerate(positions):
        for j in range(3):
            tmp_positions[i, j] -= delta

            calculator.calculate(symbols=symbols, positions=tmp_positions)

            ea = calculator.results["energy"][triger_str]
            tmp_positions[i, j] += delta * 2

            calculator.calculate(symbols=symbols, positions=tmp_positions)
            eb = calculator.results["energy"][triger_str]

            tmp_positions[i, j] -= delta
            grad[i, j] = (0.5 * eb - 0.5 * ea) / (delta)

    return grad * BOHR


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call":
        return

    if not item.config.getoption("--write-output"):
        return

    workdir = getattr(item.module, "WORKDIR", None)
    if not workdir:
        return

    source = Path(workdir)

    
    status = "failed" if report.failed else "passed"
    destination = source / f"{status}_{item.name}"
    if destination.exists():
        shutil.rmtree(destination)

    status_bis = "failed" if status=="passed" else "passed"
    destination_bis = source / f"{status_bis}_{item.name}"
    if destination_bis.exists():
        shutil.rmtree(destination_bis)

    os.makedirs(destination, exist_ok=False)

    for fichier in source.iterdir():
        if fichier.is_file():
            shutil.copy2(fichier, destination / fichier.name)
