"""Canned resolver for the ontology tests - no live msfrpcd."""
from __future__ import annotations

import pentest_ontology as o


class FakeResolver:
    def __init__(self, modules: set[str] | None = None, payloads: set[str] | None = None,
                 compatible: set[str] | None = None) -> None:
        # names stored WITHOUT type prefix, matching how validate_* normalizes
        self.modules = modules if modules is not None else {"unix/ftp/vsftpd_234_backdoor"}
        self.payloads = payloads if payloads is not None else {
            "cmd/unix/reverse_bash", "linux/x64/meterpreter/reverse_tcp"}
        self.compatible = compatible if compatible is not None else {"cmd/unix/reverse_bash"}
        self.calls: list[tuple[str, str]] = []

    async def module(self, module_type: str, name: str) -> o.ModuleInfo:
        base = name.split("/", 1)[1] if name.startswith(module_type + "/") else name
        self.calls.append(("module", base))
        exists = base in self.modules
        return o.ModuleInfo(
            exists=exists,
            required_options=("RHOSTS",) if exists else (),
            suggestions=() if exists else ("unix/ftp/vsftpd_234_backdoor",),
        )

    async def payload(self, name: str, module: str | None = None) -> o.PayloadInfo:
        base = o._norm_payload(name)
        self.calls.append(("payload", base))
        exists = base in self.payloads
        return o.PayloadInfo(
            exists=exists,
            required_options=("LHOST", "LPORT") if exists else (),
            compatible=(base in self.compatible) if (module and exists) else None,
            compatible_payloads=("cmd/unix/reverse_bash",),
            suggestions=() if exists else ("cmd/unix/generic",),
        )
