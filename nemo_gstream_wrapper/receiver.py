#!/usr/bin/env python3
import threading

import rclpy
from rclpy.node import Node

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib


class GstRxNode(Node):
    def __init__(self):
        super().__init__("gst_rx_node")

        self.declare_parameter("port", 5000)
        self.declare_parameter("latency_ms", 10)
        self.declare_parameter("sync", False)

        Gst.init(None)

        self._loop = GLib.MainLoop()
        self._pipeline = Gst.parse_launch(self._build_pipeline())
        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message", self._on_bus_message)

        self.get_logger().info("Starting RX pipeline:\n" + self._build_pipeline())

        self._gst_thread = threading.Thread(target=self._gst_main, daemon=True)
        self._gst_thread.start()

    def _build_pipeline(self) -> str:
        port = int(self.get_parameter("port").value)
        latency = int(self.get_parameter("latency_ms").value)
        sync = bool(self.get_parameter("sync").value)
        sync_str = "true" if sync else "false"

        return (
            f'udpsrc port={port} caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! '
            f'rtpjitterbuffer latency={latency} ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
            f'autovideosink sync={sync_str}'
        )

    def _gst_main(self):
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error("Failed to set RX pipeline to PLAYING")
            return
        try:
            self._loop.run()
        except Exception as e:
            self.get_logger().error(f"GLib loop exception: {e}")

    def _on_bus_message(self, bus, msg):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            self.get_logger().error(f"Gst ERROR: {err} | debug: {dbg}")
        elif t == Gst.MessageType.WARNING:
            warn, dbg = msg.parse_warning()
            self.get_logger().warn(f"Gst WARNING: {warn} | debug: {dbg}")
        elif t == Gst.MessageType.EOS:
            self.get_logger().warn("Gst EOS")

    def stop(self):
        try:
            self.get_logger().info("Stopping RX pipeline...")
        except Exception:
            pass
        try:
            if self._pipeline:
                self._pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            if self._loop and self._loop.is_running():
                self._loop.quit()
        except Exception:
            pass

    def destroy_node(self):
        self.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = GstRxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
