"""Tests for the msf_search MCP tool (tools/pentest_tools_mcp.py).

pentest_tools_mcp pulls in uvicorn/fastapi/mcp and MetasploitMCP; when those
are not importable in this interpreter the tests skip rather than fail.
The Metasploit RPC client is faked - no msfrpcd needed.

Run under pytest:   python -m pytest tools/tests
or standalone:      python tools/tests/test_msf_search.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pentest_tools_mcp as ptm
except Exception as exc:  # noqa: BLE001 - heavy optional deps
    ptm = None
    _skip_reason = str(exc)


class _FakeModules:
    def __init__(self, rows):
        self._rows = rows

    def search(self, query):
        return list(self._rows)


class _FakeClient:
    def __init__(self, rows):
        self.modules = _FakeModules(rows)


def _with_client(rows):
    ptm.msf.get_msf_client = lambda: _FakeClient(rows)


def _run(**kw):
    return asyncio.new_event_loop().run_until_complete(ptm.msf_search(**kw))


def test_empty_query_is_rejected():
    if ptm is None:
        return
    assert "pass a query" in _run(query="  ")


def test_bad_module_type_is_rejected():
    if ptm is None:
        return
    _with_client([])
    out = _run(query="x", module_type="banana")
    assert "module_type must be one of" in out


def test_no_hits_tells_the_model_to_stop_guessing():
    if ptm is None:
        return
    _with_client([])
    out = _run(query="crlf injection")
    assert "no Metasploit module" in out
    assert "do not keep guessing" in out
    assert "raw_tcp_send" in out


def test_hits_are_sorted_exploit_first_then_rank():
    if ptm is None:
        return
    _with_client([
        {"fullname": "auxiliary/scanner/http/a", "type": "auxiliary", "rank": "normal"},
        {"fullname": "exploit/x/low", "type": "exploit", "rank": "good"},
        {"fullname": "exploit/x/high", "type": "exploit", "rank": "excellent"},
        {"fullname": "", "type": "exploit", "rank": "great"},  # dropped: no name
    ])
    out = _run(query="apache")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith(("exploit/", "auxiliary/"))]
    assert lines[0].startswith("exploit/x/high")
    assert lines[1].startswith("exploit/x/low")
    assert lines[2].startswith("auxiliary/scanner/http/a")
    assert "3 module(s) match" in out


def test_module_type_filter():
    if ptm is None:
        return
    _with_client([
        {"fullname": "auxiliary/scanner/http/a", "type": "auxiliary", "rank": "normal"},
        {"fullname": "exploit/x/y", "type": "exploit", "rank": "good"},
    ])
    out = _run(query="http", module_type="auxiliary")
    assert "auxiliary/scanner/http/a" in out
    assert "exploit/x/y" not in out


def test_search_exception_is_reported_not_raised():
    if ptm is None:
        return

    class _Boom:
        modules = type("M", (), {"search": staticmethod(
            lambda q: (_ for _ in ()).throw(RuntimeError("rpc down")))})()

    ptm.msf.get_msf_client = lambda: _Boom()
    out = _run(query="apache")
    assert "failed" in out and "rpc down" in out


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
