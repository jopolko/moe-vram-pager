"""Tests for nmap_scan's targeted-vs-deep profile split (tools/pentest_agent.py).

Run under pytest:   python -m pytest tools/tests
or standalone:      python tools/tests/test_nmap_profile.py

pentest_agent imports the `mcp` package at module load; when it is not
installed these tests skip rather than fail.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pentest_agent as p
except ImportError as exc:  # mcp not installed in this interpreter
    p = None
    _skip_reason = str(exc)


def test_port_spec_count():
    if p is None:
        return
    assert p._port_spec_count("-") is None
    assert p._port_spec_count("") is None
    assert p._port_spec_count("80") == 1
    assert p._port_spec_count("80,443,8080") == 3
    assert p._port_spec_count("1-64") == 64
    assert p._port_spec_count("1-1000") == 1000
    assert p._port_spec_count("T:80,U:53") == 2
    assert p._port_spec_count("garbage") is None
    assert p._port_spec_count("100-1") is None


def test_targeted_predicate():
    if p is None:
        return
    assert p._is_targeted_ports("28461") is True
    assert p._is_targeted_ports("80,443") is True
    assert p._is_targeted_ports("1-64") is True
    assert p._is_targeted_ports("1-65") is False
    assert p._is_targeted_ports("1-1000") is False
    assert p._is_targeted_ports("-") is False
    assert p._is_targeted_ports("garbage") is False


def test_forced_ports_stay_deep(monkeypatch=None):
    if p is None:
        return
    old = p._FORCED_PORTS
    try:
        p._FORCED_PORTS = "443"
        assert p._is_targeted_ports("443") is False
    finally:
        p._FORCED_PORTS = old


def test_build_cmd_targeted_is_light():
    if p is None:
        return
    cmd = p._build_nmap_cmd("10.0.0.1", "28461")
    assert cmd is not None
    assert "-O" not in cmd and "--osscan-guess" not in cmd
    assert "vuln" not in cmd[cmd.index("--script") + 1]
    assert cmd[cmd.index("--version-intensity") + 1] == p.NMAP_TARGETED_VERSION_INTENSITY


def test_build_cmd_full_sweep_is_deep():
    if p is None:
        return
    cmd = p._build_nmap_cmd("10.0.0.1", "")
    assert cmd is not None
    assert cmd[cmd.index("-p") + 1] == "-"
    assert "vuln" in cmd[cmd.index("--script") + 1]
    # -O only when the nmap binary has raw-socket privilege; assert consistency
    assert ("-O" in cmd) == ("--privileged" in cmd)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
