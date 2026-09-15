"""Simple HTTP proxy from port 3000 to port 5173 (Vite).
Ensures anyone browsing http://localhost:3000 reaches the frontend seamlessly.
"""
import http.server
import urllib.request
import sys

TARGET = "http://127.0.0.1:5173"

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def do_HEAD(self):
        self._proxy("HEAD")

    def do_OPTIONS(self):
        self._proxy("OPTIONS")

    def _proxy(self, method):
        target_url = TARGET + self.path
        body = None
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)

        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        req = urllib.request.Request(target_url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.read())
        except Exception as e:
            self.send_response(302)
            self.send_header("Location", target_url)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Quiet logging

if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 3000), ProxyHandler)
    server.serve_forever()
