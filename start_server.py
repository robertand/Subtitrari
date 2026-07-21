#!/usr/bin/env python3
"""Helper script to start the app server"""
import sys, os, time, signal

# Start the server
pid = os.fork()
if pid == 0:
    # Child process
    sys.stdout = open('/tmp/subtitrar.log', 'w')
    sys.stderr = sys.stdout
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from app import app
    from config import Config
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True, use_reloader=False)
else:
    # Parent - print PID and exit
    print(pid)
