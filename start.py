#!/usr/bin/env python3
"""
Entry point per Railway.app e altri PaaS
"""
import os
from smallllm.api import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'false').lower() == 'true')
