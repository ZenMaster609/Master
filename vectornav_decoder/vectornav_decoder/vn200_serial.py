#!/usr/bin/env python3
"""
VN-200 Serial Communication Layer

Handles serial port communication with VectorNav VN-200.
No ROS dependencies - uses pyserial.

Features:
- Connection management with automatic reconnection
- Binary output configuration via ASCII commands
- Thread-safe read operations with callbacks
- Statistics tracking
"""

import serial
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any

from .vn200_protocol import VN200Parser, VN200Packet
from .vn200_config import (
    SerialConfig,
    BinaryOutputConfig,
    build_register_command,
    build_full_ascii_command,
    create_default_config,
)


class VN200Serial:
    """
    VN-200 serial communication handler.

    Manages:
    - Serial port open/close
    - Initial configuration (ASCII commands to set binary output)
    - Continuous binary streaming
    - Reconnection on failure

    Usage:
        config = SerialConfig(port='/dev/ttyUSB0', baudrate=921600)
        binary_config = create_default_config()

        vn = VN200Serial(config, binary_config)
        vn.set_packet_callback(my_callback)

        if vn.connect():
            vn.start_streaming()

        # ... later
        vn.stop_streaming()
        vn.disconnect()
    """

    def __init__(
        self,
        config: SerialConfig,
        binary_config: Optional[BinaryOutputConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize VN-200 serial handler.

        Args:
            config: Serial port configuration
            binary_config: Binary output configuration (optional)
            logger: Logger instance (optional)
        """
        self._config = config
        self._binary_config = binary_config or create_default_config()
        self._logger = logger or logging.getLogger(__name__)

        self._serial: Optional[serial.Serial] = None
        self._parser = VN200Parser()
        self._running = False
        self._read_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Callbacks
        self._packet_callback: Optional[Callable[[VN200Packet], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None

        # Statistics
        self._bytes_received = 0
        self._connection_count = 0
        self._last_packet_time = 0.0

    def set_packet_callback(self, callback: Callable[[VN200Packet], None]) -> None:
        """
        Set callback for received packets.

        Args:
            callback: Function called with each parsed VN200Packet
        """
        self._packet_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """
        Set callback for errors.

        Args:
            callback: Function called on serial errors
        """
        self._error_callback = callback

    def connect(self) -> bool:
        """
        Connect to VN-200 and configure binary output.

        Returns:
            True if connection and configuration successful
        """
        with self._lock:
            try:
                self._serial = serial.Serial(
                    port=self._config.port,
                    baudrate=self._config.baudrate,
                    timeout=self._config.timeout_sec,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                )
                self._connection_count += 1
                self._logger.info(
                    f"Connected to {self._config.port} at {self._config.baudrate} baud"
                )

                # Configure binary output
                if self._binary_config:
                    if not self._configure_binary_output():
                        self._logger.error("Failed to configure binary output")
                        self.disconnect()
                        return False

                return True

            except serial.SerialException as e:
                self._logger.error(f"Failed to open {self._config.port}: {e}")
                if self._error_callback:
                    self._error_callback(e)
                return False

    def disconnect(self) -> None:
        """Disconnect from VN-200."""
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self._serial = None
            self._logger.info("Disconnected from VN-200")

    def _configure_binary_output(self) -> bool:
        """
        Configure VN-200 binary output via ASCII commands.

        Sends commands to:
        1. Pause async output
        2. Configure binary output register (75, 76, or 77)
        3. Resume async output

        Returns:
            True if configuration successful
        """
        if not self._serial or not self._serial.is_open:
            return False

        try:
            # Pause async output temporarily
            self._send_ascii_command("VNASY,0")
            time.sleep(0.1)

            # Flush any pending data
            self._serial.reset_input_buffer()

            # Build and send configuration command
            config_cmd = build_register_command(self._binary_config)
            self._logger.info(f"Configuring binary output: ${config_cmd}")

            response = self._send_ascii_command(config_cmd)
            if response and 'ERR' in response:
                self._logger.error(f"Configuration error: {response}")
                return False

            time.sleep(0.1)

            # Resume async output
            self._send_ascii_command("VNASY,1")
            time.sleep(0.1)

            self._logger.info(
                f"Binary output configured: {self._binary_config.output_rate_hz:.0f} Hz"
            )
            return True

        except Exception as e:
            self._logger.error(f"Configuration failed: {e}")
            return False

    def _send_ascii_command(self, command: str) -> Optional[str]:
        """
        Send ASCII command and wait for response.

        Args:
            command: Command without $ prefix or checksum

        Returns:
            Response string or None on timeout
        """
        if not self._serial or not self._serial.is_open:
            return None

        try:
            full_cmd = build_full_ascii_command(command)
            self._serial.write(full_cmd)

            # Read response (may timeout)
            response = self._serial.readline()
            if response:
                return response.decode('ascii', errors='ignore').strip()
            return None

        except Exception as e:
            self._logger.warning(f"ASCII command failed: {e}")
            return None

    def start_streaming(self) -> None:
        """Start background thread for reading binary data."""
        if self._running:
            return

        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        self._logger.info("Started streaming thread")

    def stop_streaming(self) -> None:
        """Stop background read thread."""
        self._running = False
        if self._read_thread:
            self._read_thread.join(timeout=2.0)
            self._read_thread = None
        self._logger.info("Stopped streaming thread")

    def _read_loop(self) -> None:
        """Background thread for reading serial data."""
        reconnect_attempts = 0

        while self._running:
            try:
                # Check connection
                if not self._serial or not self._serial.is_open:
                    # Attempt reconnection
                    if reconnect_attempts >= self._config.max_reconnect_attempts:
                        self._logger.error("Max reconnection attempts reached")
                        break

                    self._logger.info(
                        f"Reconnecting... (attempt {reconnect_attempts + 1})"
                    )
                    time.sleep(self._config.reconnect_delay_sec)

                    if self.connect():
                        reconnect_attempts = 0
                        self._parser.reset()
                    else:
                        reconnect_attempts += 1
                    continue

                # Read available data
                with self._lock:
                    if self._serial and self._serial.is_open:
                        # Read up to 1024 bytes (non-blocking with timeout)
                        data = self._serial.read(min(1024, self._serial.in_waiting or 1))
                    else:
                        data = b''

                if data:
                    self._bytes_received += len(data)
                    packets = self._parser.feed(data)

                    for packet in packets:
                        self._last_packet_time = time.time()
                        packet.timestamp_ns = int(time.time() * 1e9)

                        if self._packet_callback:
                            try:
                                self._packet_callback(packet)
                            except Exception as e:
                                self._logger.warning(f"Packet callback error: {e}")

            except serial.SerialException as e:
                self._logger.warning(f"Serial error: {e}")
                self.disconnect()
                if self._error_callback:
                    self._error_callback(e)

            except Exception as e:
                self._logger.error(f"Read loop error: {e}")

    def send_command(self, command: str) -> Optional[str]:
        """
        Send an ASCII command (for advanced users).

        Args:
            command: Command without $ prefix or checksum

        Returns:
            Response string or None
        """
        with self._lock:
            return self._send_ascii_command(command)

    @property
    def is_connected(self) -> bool:
        """Check if serial port is open."""
        with self._lock:
            return self._serial is not None and self._serial.is_open

    @property
    def is_streaming(self) -> bool:
        """Check if streaming thread is running."""
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        """Return communication statistics."""
        return {
            'bytes_received': self._bytes_received,
            'connection_count': self._connection_count,
            'last_packet_time': self._last_packet_time,
            'is_connected': self.is_connected,
            'is_streaming': self.is_streaming,
            'parser_stats': self._parser.stats,
        }

    def get_time_since_last_packet(self) -> float:
        """
        Get time since last packet was received.

        Returns:
            Time in seconds, or float('inf') if no packets received
        """
        if self._last_packet_time == 0:
            return float('inf')
        return time.time() - self._last_packet_time
