
import pytest



def pytest_addoption(parser):
    parser.addoption(
        "--run-optional",
        action="store_true",
        default=False,
        help="run optional tests"
    )
    parser.addoption(
        "--beta",
        action="store_true",
        default=False,
        help="run beta-features tests"
    )
    parser.addoption(
        "--bird",
        action="store_true",
        default=False,
        help="run bird-features tests"
    )
    parser.addoption(
        "--dftbplus",
        action="store_true",
        default=False,
        help="run dftbplus-features tests"
    )
    parser.addoption(
        "--forces",
        action="store_true",
        default=False,
        help="run forces-features tests"
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

    if not config.getoption("--bird"):
        skip_optional = pytest.mark.skip(reason="need --bird option")
        for item in items:
            if "bird" in item.keywords:
                item.add_marker(skip_optional)

    if not config.getoption("--dftbplus"):
        skip_optional = pytest.mark.skip(reason="need --dftbplus option")
        for item in items:
            if "dftbplus" in item.keywords:
                item.add_marker(skip_optional)

    if not config.getoption("--forces"):
        skip_optional = pytest.mark.skip(reason="need --forces option")
        for item in items:
            if "forces" in item.keywords:
                item.add_marker(skip_optional)



import numpy as np

BOHR = 0.529177210544

def compute_numgrad(symbols, positions, calculator, delta=0.01):
    
    grad = np.zeros(np.shape(positions))

    tmp_positions = positions.copy()

    for i,atm in enumerate(positions):
        for j in range(3):
            
            tmp_positions[i,j] -= delta
            
            calculator.calculate(
                symbols=symbols,
                positions=tmp_positions
            )
            
            ea = calculator.results["energy"]["energy"]
            tmp_positions[i,j] += delta * 2

            calculator.calculate(
                symbols=symbols,
                positions=tmp_positions
            )
            eb = calculator.results["energy"]["energy"]

            tmp_positions[i,j] -= delta 
            grad[i,j] = (0.5*eb - 0.5*ea) / (delta)
    
    return grad * BOHR




