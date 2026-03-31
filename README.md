# **ROS 2 GStreamer Low-Latency Video Streaming (TX + RX)**

A lightweight ROS 2 package for low-latency H.264 video streaming over UDP using GStreamer.
This package provides two nodes:
1. **gst_tx_node** — Captures from V4L2 camera → encodes H.264 → sends RTP over UDP
2. **gst_rx_node** — Receives RTP H.264 over UDP → decodes → displays videohis i

### This is the code with aruco detection for the bottom camera

### Running the System

**Step 1 — Start Receiver (Host Machine)**

```bash
ros2 run nemo_gstream_wrapper receiver
```

**Step 2 — Start Transmitter (Camera Device)**

```bash
ros2 run gst_stream_pkg transmitter
```

Designed for:
Embedded vision systems
Edge devices (Jetson / ARM / x86)
Ethernet-based streaming
Robotics / remote perception pipelines
Low-latency video transport

## System Architecture

Transmitter (TX)
Camera → H.264 Encoder → RTP Packetizer → UDP

Default pipeline:
```
v4l2src device=/dev/video0 !
video/x-raw,width=640,height=480,framerate=30/1 !
videoconvert !
x264enc tune=zerolatency speed-preset=veryfast bitrate=5000 key-int-max=60 bframes=0 !
rtph264pay config-interval=1 pt=96 !
udpsink host=192.168.2.1 port=5000
```

## Receiver (RX)


UDP → RTP Depacketizer → H.264 Decode → Display

Default pipeline:
```
udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" !
rtpjitterbuffer latency=10 !
rtph264depay !
h264parse !
avdec_h264 !
videoconvert !
autovideosink sync=false
Requirements
System Dependencies (Ubuntu / Debian)
sudo apt update
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  python3-gi \
  gir1.2-gstreamer-1.0 \
  gir1.2-gobject-2.0 \
  v4l-utils
```
## Verify installation:

`gst-launch-1.0 --version`

### Workspace Structure Example
```ros2_ws/
  src/
    gst_stream_pkg/
      package.xml
      setup.py
      gst_stream_pkg/
        gst_tx_node.py
        gst_rx_node.py
```

## Build Instructions

```
cd ros2_ws
colcon build --packages-select gst_stream_pkg
source install/setup.bash
```
