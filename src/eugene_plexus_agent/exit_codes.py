"""What a Windows exit status means, in words a person can act on.

A process Windows itself refuses to run, or ends, exits with an NTSTATUS
rather than a code of its own: 3221225781 is `0xC0000135`, *a DLL it
needs was not found*. The supervisor used to report that as "exited with
code 3221225781".

**Found 2026-09-26 on a friend's freshly reinstalled Windows.** llama.cpp
installed, every model "crashed", and nothing on any screen said why. The
Windows build of `llama-server` imports `VCRUNTIME140.dll`,
`VCRUNTIME140_1.dll` and `MSVCP140.dll`. It does not ship them; they come
from the Microsoft Visual C++ Redistributable, which a new Windows does not
have. Eugene's own Python carries its own copy, so everything else worked.

These are the statuses a child of this agent can meet at start or under
load. Anything else is named by its code, so the next reader can look it
up.
"""

from __future__ import annotations

VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

# NTSTATUS values, as unsigned 32-bit numbers.
_DLL_NOT_FOUND = 0xC0000135
_ENTRYPOINT_NOT_FOUND = 0xC0000139
_INVALID_IMAGE_FORMAT = 0xC000007B
_ILLEGAL_INSTRUCTION = 0xC000001D
_ACCESS_VIOLATION = 0xC0000005
_NO_MEMORY = 0xC0000017
_COMMITMENT_LIMIT = 0xC000012D
_STACK_BUFFER_OVERRUN = 0xC0000409
_CONTROL_C_EXIT = 0xC000013A


def explain_windows_exit(return_code: int) -> str | None:
    """The cause and the next step for a Windows NTSTATUS exit, or None.

    `return_code` may arrive signed or unsigned depending on who read it,
    so it is normalised to 32 bits first. A code below `0xC0000000` is the
    program's own and says nothing Windows-specific.
    """
    code = return_code & 0xFFFFFFFF
    if code < 0xC0000000:
        return None
    if code == _DLL_NOT_FOUND:
        return (
            "it could not start: Windows could not find a DLL it needs "
            "(0xC0000135). For llama.cpp that is the Microsoft Visual C++ "
            "Redistributable (x64), which a freshly installed Windows does not "
            f"have. Install it from {VC_REDIST_URL}, then press start."
        )
    if code == _ENTRYPOINT_NOT_FOUND:
        return (
            "it could not start: a DLL it loaded is too old for it (0xC0000139). "
            "Update the Microsoft Visual C++ Redistributable (x64) from "
            f"{VC_REDIST_URL}, then press start."
        )
    if code == _INVALID_IMAGE_FORMAT:
        return (
            "it could not start: a DLL it loaded was built for a different kind "
            "of processor (0xC000007B). Reinstall the engine from the Inference "
            "page; if it happens again, reinstall the Microsoft Visual C++ "
            f"Redistributable (x64) from {VC_REDIST_URL}."
        )
    if code == _ILLEGAL_INSTRUCTION:
        return (
            "this processor lacks an instruction the build uses (0xC000001D). "
            "In a virtual machine, pass the host's processor features through to "
            "it; otherwise this build cannot run on this processor."
        )
    if code in (_NO_MEMORY, _COMMITMENT_LIMIT):
        return (
            f"Windows ran out of memory for it (0x{code:08X}). Close other "
            "programs, choose a smaller model or context, or let Windows grow "
            "its page file."
        )
    if code == _ACCESS_VIOLATION:
        return (
            "it crashed (access violation, 0xC0000005). Its last lines in the "
            "agent's log say where; a smaller context or a different build may "
            "avoid it."
        )
    if code == _STACK_BUFFER_OVERRUN:
        return (
            "it stopped itself on a fatal error (0xC0000409). Its last lines in "
            "the agent's log say why."
        )
    if code == _CONTROL_C_EXIT:
        return "it was interrupted (0xC000013A), as if Ctrl+C had been pressed in it."
    return f"Windows ended it with status 0x{code:08X}."


__all__ = ["VC_REDIST_URL", "explain_windows_exit"]
