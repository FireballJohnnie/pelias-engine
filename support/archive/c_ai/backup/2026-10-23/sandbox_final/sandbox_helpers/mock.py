import json,http.server,time
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        n=int(self.headers['Content-Length']); b=json.loads(self.rfile.read(n))
        if self.path=='/tokenize':
            out=json.dumps({"tokens":list(range(len(b["content"].split())))}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(out); return
        self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
        for i in range(b["n_predict"]):
            self.wfile.write(("data: "+json.dumps({"content":"x","stop":False})+"\n\n").encode()); self.wfile.flush(); time.sleep(0.002)
        self.wfile.write(("data: "+json.dumps({"content":"","stop":True,"tokens_evaluated":123})+"\n\n").encode())
http.server.ThreadingHTTPServer(('127.0.0.1',18012),H).serve_forever()
