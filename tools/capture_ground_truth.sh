#!/bin/bash
#
# Capture ROS 2 system ground truth for debugging and documentation
#
# Saves:
#   - nodes.txt: Active nodes
#   - topics.txt: Topics with message types
#   - system_info.json: Hostname, IPs, environment
#
# Usage:
#   ./capture_ground_truth.sh                    # Output to ./ground_truth_<timestamp>/
#   ./capture_ground_truth.sh /path/to/output    # Output to specified directory
#

set -e

# Determine output directory
if [ -n "$1" ]; then
    OUTPUT_DIR="$1"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="./ground_truth_${TIMESTAMP}"
fi

echo "Capturing ground truth to: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Capture node list
echo "Capturing nodes..."
ros2 node list > "$OUTPUT_DIR/nodes.txt" 2>&1 || echo "No nodes found" > "$OUTPUT_DIR/nodes.txt"

# Capture topic list with types
echo "Capturing topics..."
ros2 topic list -t > "$OUTPUT_DIR/topics.txt" 2>&1 || echo "No topics found" > "$OUTPUT_DIR/topics.txt"

# Capture node info for each node
echo "Capturing node details..."
NODE_INFO_DIR="$OUTPUT_DIR/node_info"
mkdir -p "$NODE_INFO_DIR"

while IFS= read -r node; do
    if [ -n "$node" ]; then
        safe_name=$(echo "$node" | tr '/' '_' | sed 's/^_//')
        ros2 node info "$node" > "$NODE_INFO_DIR/${safe_name}.txt" 2>&1 || true
    fi
done < "$OUTPUT_DIR/nodes.txt"

# Capture system info as JSON
echo "Capturing system info..."
cat > "$OUTPUT_DIR/system_info.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "user": "$USER",
    "platform": "$(uname -s)",
    "kernel": "$(uname -r)",
    "architecture": "$(uname -m)",
    "ros_distro": "${ROS_DISTRO:-unknown}",
    "ros_domain_id": "${ROS_DOMAIN_ID:-0}",
    "rmw_implementation": "${RMW_IMPLEMENTATION:-default}",
    "cyclonedds_uri": "${CYCLONEDDS_URI:-}",
    "fastrtps_profile": "${FASTRTPS_DEFAULT_PROFILES_FILE:-}",
    "pwd": "$(pwd)",
    "network": {
        "interfaces": [
$(ip -4 addr show 2>/dev/null | grep -E "inet |^[0-9]" | awk '
    /^[0-9]/ { iface=$2; gsub(/:/, "", iface) }
    /inet / { print "            \"" iface "\": \"" $2 "\"," }
' | sed '$ s/,$//')
        ]
    }
}
EOF

# Capture topic frequencies (quick sample)
echo "Sampling topic frequencies..."
FREQ_FILE="$OUTPUT_DIR/topic_frequencies.txt"
echo "# Topic frequency samples (1 second each)" > "$FREQ_FILE"
echo "# Captured at $(date -Iseconds)" >> "$FREQ_FILE"
echo "" >> "$FREQ_FILE"

# Sample a few key topics
for topic in /vehicle_plotter/state /run_session /vectornav/imu /can/wheel_velocities; do
    if ros2 topic list | grep -q "^${topic}$"; then
        echo "Sampling $topic..."
        echo "## $topic" >> "$FREQ_FILE"
        timeout 2 ros2 topic hz "$topic" 2>&1 | head -5 >> "$FREQ_FILE" || echo "  No data" >> "$FREQ_FILE"
        echo "" >> "$FREQ_FILE"
    fi
done

# Summary
echo ""
echo "========================================="
echo "Ground truth captured to: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -la "$OUTPUT_DIR"
echo ""
echo "Node count: $(wc -l < "$OUTPUT_DIR/nodes.txt")"
echo "Topic count: $(wc -l < "$OUTPUT_DIR/topics.txt")"
echo "========================================="
