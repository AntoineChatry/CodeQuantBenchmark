"""Mesure mémoire du processus courant (Windows via ctypes, fallback cross-platform)."""

import ctypes
import ctypes.wintypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MemorySnapshot:
    rss_bytes: int
    rss_mb: float
    peak_rss_bytes: int
    peak_rss_mb: float


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_memory_windows() -> MemorySnapshot:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    # K32GetProcessMemoryInfo is in kernel32 on Windows 7+ and avoids psapi issues
    try:
        K32GetProcessMemoryInfo = kernel32.K32GetProcessMemoryInfo
    except AttributeError:
        # Fallback to psapi on older systems
        psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
        K32GetProcessMemoryInfo = psapi.GetProcessMemoryInfo

    handle = kernel32.GetCurrentProcess()
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)

    K32GetProcessMemoryInfo.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        ctypes.wintypes.DWORD,
    ]
    K32GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL

    success = K32GetProcessMemoryInfo(
        handle, ctypes.byref(counters), ctypes.sizeof(counters)
    )
    if not success:
        raise OSError(
            f"K32GetProcessMemoryInfo failed (error {ctypes.get_last_error()})"
        )

    return MemorySnapshot(
        rss_bytes=counters.WorkingSetSize,
        rss_mb=round(counters.WorkingSetSize / (1024 * 1024), 2),
        peak_rss_bytes=counters.PeakWorkingSetSize,
        peak_rss_mb=round(counters.PeakWorkingSetSize / (1024 * 1024), 2),
    )


def get_memory_unix() -> MemorySnapshot:
    try:
        with open("/proc/self/status", "r") as f:
            lines = f.readlines()
        vm_rss = 0
        vm_peak = 0
        for line in lines:
            if line.startswith("VmRSS:"):
                vm_rss = int(line.split()[1]) * 1024
            elif line.startswith("VmPeak:"):
                vm_peak = int(line.split()[1]) * 1024
        return MemorySnapshot(
            rss_bytes=vm_rss,
            rss_mb=round(vm_rss / (1024 * 1024), 2),
            peak_rss_bytes=vm_peak,
            peak_rss_mb=round(vm_peak / (1024 * 1024), 2),
        )
    except FileNotFoundError:
        return MemorySnapshot(rss_bytes=0, rss_mb=0.0, peak_rss_bytes=0, peak_rss_mb=0.0)


def get_memory() -> MemorySnapshot:
    if sys.platform == "win32":
        return get_memory_windows()
    return get_memory_unix()


def get_memory_of_pid(pid: int) -> MemorySnapshot:
    """Measure RSS of an external process by PID."""
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        try:
            K32GetProcessMemoryInfo = kernel32.K32GetProcessMemoryInfo
        except AttributeError:
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            K32GetProcessMemoryInfo = psapi.GetProcessMemoryInfo

        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        kernel32.OpenProcess.argtypes = [
            ctypes.wintypes.DWORD,
            ctypes.wintypes.BOOL,
            ctypes.wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
        kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
        )
        if not handle:
            return MemorySnapshot(rss_bytes=0, rss_mb=0.0, peak_rss_bytes=0, peak_rss_mb=0.0)

        try:
            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            K32GetProcessMemoryInfo.argtypes = [
                ctypes.wintypes.HANDLE,
                ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                ctypes.wintypes.DWORD,
            ]
            K32GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
            success = K32GetProcessMemoryInfo(
                handle, ctypes.byref(counters), ctypes.sizeof(counters)
            )
            if not success:
                return MemorySnapshot(rss_bytes=0, rss_mb=0.0, peak_rss_bytes=0, peak_rss_mb=0.0)
            return MemorySnapshot(
                rss_bytes=counters.WorkingSetSize,
                rss_mb=round(counters.WorkingSetSize / (1024 * 1024), 2),
                peak_rss_bytes=counters.PeakWorkingSetSize,
                peak_rss_mb=round(counters.PeakWorkingSetSize / (1024 * 1024), 2),
            )
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            with open(f"/proc/{pid}/status", "r") as f:
                lines = f.readlines()
            vm_rss = 0
            vm_peak = 0
            for line in lines:
                if line.startswith("VmRSS:"):
                    vm_rss = int(line.split()[1]) * 1024
                elif line.startswith("VmPeak:"):
                    vm_peak = int(line.split()[1]) * 1024
            return MemorySnapshot(
                rss_bytes=vm_rss,
                rss_mb=round(vm_rss / (1024 * 1024), 2),
                peak_rss_bytes=vm_peak,
                peak_rss_mb=round(vm_peak / (1024 * 1024), 2),
            )
        except FileNotFoundError:
            return MemorySnapshot(rss_bytes=0, rss_mb=0.0, peak_rss_bytes=0, peak_rss_mb=0.0)


def get_file_size(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"file_size_bytes": 0, "file_size_mb": 0.0, "file_size_gb": 0.0}
    size = os.path.getsize(path)
    return {
        "file_size_bytes": size,
        "file_size_mb": round(size / (1024 * 1024), 4),
        "file_size_gb": round(size / (1024 * 1024 * 1024), 6),
    }
