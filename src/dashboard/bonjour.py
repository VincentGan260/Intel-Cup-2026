"""Advertise the local RiderGuardian dashboard over Bonjour/mDNS."""

from __future__ import annotations

import socket
from dataclasses import dataclass


def _local_ipv4_address() -> str:
    """Return the preferred LAN address without sending application data."""
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
            self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
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
