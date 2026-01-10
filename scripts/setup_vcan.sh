#!/bin/bash
#
# Setup virtual CAN interface for testing
#
# This script creates a virtual CAN (vcan) interface that can be used
# to test the CAN decoder without real hardware.
#
# Usage:
#   ./setup_vcan.sh         # Setup vcan0
#   ./setup_vcan.sh vcan1   # Setup custom interface
#

set -e

INTERFACE="${1:-vcan0}"

echo "Setting up virtual CAN interface: $INTERFACE"

# Load vcan kernel module
if ! lsmod | grep -q "^vcan"; then
    echo "Loading vcan kernel module..."
    sudo modprobe vcan
fi

# Check if interface already exists
if ip link show "$INTERFACE" &>/dev/null; then
    echo "Interface $INTERFACE already exists"

    # Check if it's up
    if ip link show "$INTERFACE" | grep -q "UP"; then
        echo "$INTERFACE is already up and running"
    else
        echo "Bringing up $INTERFACE..."
        sudo ip link set up "$INTERFACE"
    fi
else
    echo "Creating $INTERFACE..."
    sudo ip link add dev "$INTERFACE" type vcan
    sudo ip link set up "$INTERFACE"
fi

# Verify
echo ""
echo "Interface status:"
ip -details link show "$INTERFACE"

echo ""
echo "========================================="
echo "$INTERFACE is ready for use!"
echo ""
echo "To test with the vehicle plotter:"
echo ""
echo "  Terminal 1 - Start vcan publisher:"
echo "    ros2 run canbus_decoder vcan_publisher_node --ros-args -p mode:=circle"
echo ""
echo "  Terminal 2 - Start CAN decoder:"
echo "    ros2 run canbus_decoder can_decoder_node"
echo ""
echo "  Terminal 3 - Or launch the full test stack:"
echo "    ros2 launch vehicle_plotter vcan_test.launch.py"
echo ""
echo "To send test frames manually:"
echo "  cansend $INTERFACE 0B9#0A001400323232"
echo "  cansend $INTERFACE 0BA#1E00280032320000"
echo ""
echo "To monitor raw CAN traffic:"
echo "  candump $INTERFACE"
echo "========================================="
