#!/usr/bin/env python3
"""Run our probes (bench/, e2e-battery, concurrency-guards) unchanged against pico's dsfront.

Listens on 127.0.0.1:SHIM_PORT (default 18894) and forwards to dsfront on 127.0.0.1:UP_PORT
(default 18893; with UP_TLS=1 any HTTPS front, such as :8888), adding "Authorization: Bearer <key>" from KEY_FILE (default: the card key that stage.sh writes).
GET /metrics (vLLM/Flare probes read it; dsfront has no such route) answers 200 with an empty body.
Responses pass through byte for byte, streamed as they arrive. Standard library only.
"""
import http.client, http.server, os, socketserver

UP_HOST = os.environ.get("UP_HOST", "127.0.0.1")
UP_PORT = int(os.environ.get("UP_PORT", "18893"))
UP_TLS = os.environ.get("UP_TLS", "0") == "1"  # UP_HOST=<head name in the certificate> UP_PORT=8888 UP_TLS=1: through the TLS front
PORT = int(os.environ.get("SHIM_PORT", "18894"))
KEY = open(os.path.expanduser(os.environ.get("KEY_FILE", "/etc/spark-llm/profiles/deepseek-v41-1017-pico.key"))).read().strip()
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization"}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def forward(self, method):
        if method == "GET" and self.path.split("?")[0] == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        head = {k: v for k, v in self.headers.items() if k.lower() not in HOP | {"host", "authorization"}}
        head["Authorization"] = "Bearer " + KEY
        up = (http.client.HTTPSConnection(UP_HOST, UP_PORT, timeout=3600) if UP_TLS
              else http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=3600))
        try:
            up.request(method, self.path, body=body, headers=head)
            r = up.getresponse()
            self.send_response(r.status, r.reason)
            for k, v in r.getheaders():
                if k.lower() not in HOP and k.lower() != "content-length":
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        finally:
            up.close()

    def do_GET(self):
        self.forward("GET")

    def do_POST(self):
        self.forward("POST")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    Server(("127.0.0.1", PORT), Handler).serve_forever()
