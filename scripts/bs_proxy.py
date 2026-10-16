"""让 BaoStock（直连 TCP 10030 端口）经由 HTTPS_PROXY 的 CONNECT 隧道连接。import 后再调用 baostock.login()。"""
import os
import socket

import baostock.util.socketutil as _su

_orig = socket.socket
_h, _p = os.environ["HTTPS_PROXY"].replace("http://", "").rstrip("/").split(":")


class _Tunnel(_orig):
    def connect(self, addr):
        host, port = addr
        _orig.connect(self, (_h, int(_p)))
        self.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.recv(1)
            if not chunk:
                raise OSError("proxy closed")
            buf += chunk
        if b" 200 " not in buf.split(b"\r\n")[0]:
            raise OSError(buf[:100])


class _SockMod:
    def __getattr__(self, k):
        return getattr(socket, k)

    socket = _Tunnel


_su.socket = _SockMod()
