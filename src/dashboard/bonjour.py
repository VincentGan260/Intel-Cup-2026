"""Advertise the local RiderGuardian dashboard over Bonjour/mDNS."""

from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass


def _local_ipv4_address() -> str:
    """Return the IPv4 address of the default physical LAN interface."""
    try:
        routes = subprocess.run(
            ["ip", "-4", "route", "show", "default"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.splitlines()
        for route in routes:
            fields = route.split()
            if "dev" not in fields:
                continue
            interface = fields[fields.index("dev") + 1]
            if interface.lower() in {"lo", "meta"}:
                continue
            addresses = subprocess.run(
                ["ip", "-4", "-o", "addr", "show", "dev", interface, "scope", "global"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.splitlines()
            for entry in addresses:
                fields = entry.split()
                if "inet" in fields:
                    return fields[fields.index("inet") + 1].split("/", 1)[0]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass

    # Portable fallback for systems without iproute2.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


@dataclass
class BonjourDashboardService:
    port: int
    device_id: str

    def __post_init__(self) -> None:
        self._zeroconf = None
        self._info = None

    def start(self) -> bool:
        try:
            from zeroconf import IPVersion, ServiceInfo, Zeroconf
        except ImportError:
            print("[Bonjour] zeroconf is not installed; local discovery is disabled")
            return False

        try:
            address = _local_ipv4_address()
            host = socket.gethostname().removesuffix(".local")
            service_type = "_riderguardian._tcp.local."
            service_name = f"RiderGuardian-{self.device_id}.{service_type}"
            self._info = ServiceInfo(
                type_=service_type,
                name=service_name,
                addresses=[socket.inet_aton(address)],
                port=self.port,
                properties={
                    b"device_id": self.device_id.encode("utf-8"),
                    b"path": b"/",
                },
                server=f"{host}.local.",
            )
            self._zeroconf = Zeroconf(
                interfaces=[address],
                ip_version=IPVersion.V4Only,
            )
            self._zeroconf.register_service(self._info)
            print(f"[Bonjour] {service_name} -> http://{address}:{self.port}")
            return True
        except Exception as exc:
            print(f"[Bonjour] unable to publish local dashboard: {exc}")
            self.close()
            return False

    def close(self) -> None:
        if self._zeroconf is None:
            return
        try:
            if self._info is not None:
                self._zeroconf.unregister_service(self._info)
        finally:
            self._zeroconf.close()
            self._zeroconf = None
            self._info = None
