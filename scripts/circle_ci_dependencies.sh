#!/bin/bash -ef

echo "Working around PyQt5 bugs"
# https://github.com/ContinuumIO/anaconda-issues/issues/9190#issuecomment-386508136
# https://github.com/golemfactory/golem/issues/1019
sudo apt update
sudo apt install libosmesa6 libglx-mesa0 libopengl0 libglx0 libdbus-1-3 \
	libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
	libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 \
	graphviz optipng
if [ ! -f /usr/lib/x86_64-linux-gnu/libxcb-util.so.1 ]; then
	sudo ln -s /usr/lib/x86_64-linux-gnu/libxcb-util.so.0 /usr/lib/x86_64-linux-gnu/libxcb-util.so.1
fi
