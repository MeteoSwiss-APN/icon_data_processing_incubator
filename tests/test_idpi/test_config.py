# First-party
import idpi.config


def test_config():
    idpi.config.set(opt1=1)
    assert idpi.config.get("opt1") == 1

    with idpi.config.set(opt1=False, opt2=2):
        assert not idpi.config.get("opt1")

    assert idpi.config.get("opt1") == 1
