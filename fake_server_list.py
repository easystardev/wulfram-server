#!/usr/bin/env python3
"""
Fake Wulfram server list HTTP server.

The client fetches http://www.wulfram.com/server_info.php to get the server list.
We serve a fake response pointing to localhost.

To use:
1. Add to hosts file: 127.0.0.1 www.wulfram.com wulfram.com
2. Run this on port 80 (requires admin/root)
3. Run wulfram_server.py
4. Launch the game
"""

import http.server
import socketserver

PORT = 8080  # Use unprivileged port (no admin needed)
GAME_PORT = 2627  # Must match wulfram_server.py

# Plain text format - what the game actually expects
# Format: line 1 = server count, then each line: name|ip|port|players|maxplayers
FAKE_SERVER_LIST_TEXT = f"""1
Wulfram Revival|127.0.0.1|{GAME_PORT}|0|50
"""

# HTML format for browsers (humans viewing the page)
FAKE_SERVER_LIST_HTML = f"""<!DOCTYPE html>
<html>
<head>
    <title>Wulfram 2 Server List</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #0f0; }}
        .server {{ background: #16213e; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .server a {{ color: #0ff; font-size: 18px; text-decoration: none; }}
        .server a:hover {{ text-decoration: underline; }}
        .info {{ color: #888; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>Wulfram 2 Server Browser</h1>
    <div class="server">
        <a href="/connect.w2l" download="connect.w2l">Wulfram Revival (Local)</a>
        <div class="info">Players: 0/50 | IP: 127.0.0.1:{GAME_PORT}</div>
    </div>
    <p style="color:#666; margin-top:30px;">Click a server name to download connection file, then open it.</p>
</body>
</html>
"""


class WulframHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f"Request: {self.path}")
        print(f"  User-Agent: {self.headers.get('User-Agent', 'unknown')}")

        if 'server_info' in self.path:
            # Detect if request is from a browser or the game
            user_agent = self.headers.get('User-Agent', '').lower()
            is_browser = any(x in user_agent for x in ['mozilla', 'chrome', 'safari', 'edge', 'firefox'])

            if is_browser:
                # Serve HTML for human viewing
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(FAKE_SERVER_LIST_HTML.encode('utf-8'))
                print(f"Served fake server list (HTML for browser)")
            else:
                # Serve plain text for the game client
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(FAKE_SERVER_LIST_TEXT.encode('utf-8'))
                print(f"Served fake server list (plain text for game)")
        elif 'updates' in self.path:
            # No updates available
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"0\n")
            print(f"Served fake updates (none)")
        elif 'connect.w2l' in self.path:
            # Serve .w2l connection file for download
            w2l_content = f"127.0.0.1\r\n{GAME_PORT}\r\n"
            self.send_response(200)
            self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Content-Disposition', 'attachment; filename="connect.w2l"')
            self.end_headers()
            self.wfile.write(w2l_content.encode('utf-8'))
            print(f"Served connect.w2l file for download")
        else:
            # Serve HTML page that launches the game via protocol
            launch_html = f'''<!DOCTYPE html>
<html>
<head><title>Launching Wulfram 2...</title></head>
<body style="background:#1a1a2e;color:#eee;font-family:Arial;text-align:center;padding-top:100px;">
<h1>Launching Wulfram 2...</h1>
<p>If the game doesn't open automatically, <a href="wulfram2://127.0.0.1:{GAME_PORT}" style="color:#0ff;">click here</a></p>
<script>window.location.href = "wulfram2://127.0.0.1:{GAME_PORT}";</script>
</body>
</html>'''
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(launch_html.encode('utf-8'))
            print(f"Served launch page for wulfram2:// protocol")


def main():
    print(f"Starting fake Wulfram HTTP server on port {PORT}")
    print(f"Game server should be running on port {GAME_PORT}")
    print()
    print(f"Make sure client_params has:")
    print(f'  www_launch "http://127.0.0.1:{PORT}/server_info.php"')
    print()

    with socketserver.TCPServer(("", PORT), WulframHTTPHandler) as httpd:
        print(f"Listening on http://127.0.0.1:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == '__main__':
    main()
