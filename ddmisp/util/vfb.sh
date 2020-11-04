#!/bin/bash

export DISPLAY=:99.0
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 1
export MESA_GL_VERSION_OVERRIDE=3.3
exec "$@"

