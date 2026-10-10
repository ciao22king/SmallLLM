#!/usr/bin/env python3
"""
Simple HTTP server to serve SmallLLM web interface
"""

import http.server
import socketserver
import os
import sys
import webbrowser
import threading

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        # Serve index.html for root path
        if self.path == '/' or self.path == '/index.html':
            self.path = '/index.html'
        
        # Try to serve the file
        try:
            return super().do_GET()
        except FileNotFoundError:
            # If file not found, serve index.html (for SPA-like behavior)
            self.path = '/index.html'
            return super().do_GET()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass


def run_server(port=PORT):
    """Run the HTTP server."""
    print(f"Starting SmallLLM Web Interface...")
    print(f"Serving files from: {DIRECTORY}")
    print(f"Server running at: http://localhost:{port}")
    print()
    print("To use SmallLLM:")
    print(f"1. Open your browser and go to: http://localhost:{port}")
    print("2. Make sure SmallLLM API is running:")
    print("   python -m smallllm.api")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    
    # Open browser
    webbrowser.open(f'http://localhost:{port}')
    
    # Start server
    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()


if __name__ == '__main__':
    # Parse command line arguments
    port = PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            sys.exit(1)
    
    run_server(port)
