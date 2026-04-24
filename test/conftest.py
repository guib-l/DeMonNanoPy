
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







