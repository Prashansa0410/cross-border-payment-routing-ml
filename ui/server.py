import http.server
import socketserver
import os
import sys

DEFAULT_PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class ShowcaseHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        # Route /demo or /demo/ to demo.html
        if self.path in ('/demo', '/demo/'):
            self.path = '/demo.html'
        # Route / or /index to index.html
        elif self.path in ('/', '/index'):
            self.path = '/index.html'
        
        return super().do_GET()

def find_available_server(start_port=3000, max_attempts=10):
    for offset in range(max_attempts):
        port = start_port + offset
        try:
            socketserver.TCPServer.allow_reuse_address = True
            server = socketserver.TCPServer(("", port), ShowcaseHandler)
            return server, port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an open port starting from {start_port}.")

def run_server():
    os.chdir(DIRECTORY)
    start_port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    httpd, port = find_available_server(start_port)

    print("=" * 65)
    print("🚀 RouteFlow AI — LinkedIn Showcase Web Server Running")
    print("=" * 65)
    print(f"📖 Layman Visual Funnel:     http://localhost:{port}/demo")
    print(f"⚡ Full Technical Dashboard:   http://localhost:{port}/")
    print("=" * 65)
    print("Press Ctrl+C to stop the server.")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down showcase server.")

if __name__ == "__main__":
    run_server()
