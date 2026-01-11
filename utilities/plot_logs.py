import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from pathlib import Path

# Change session name if needed
session = "session_20260109_095334"

path = (
    Path.home()
    / "jetson_ros_logs"
    / session
    / "vehicle_state_0000.parquet"
)

table = pq.read_table(path)
df = table.to_pandas()

# Plot trajectory
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(df["x"], df["y"])
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Trajectory")

plt.subplot(2, 2, 2)
plt.plot(df["timestamp"], df["speed"])
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Speed")

plt.subplot(2, 2, 3)
plt.plot(df["timestamp"], df["yaw"])
plt.xlabel("Time (s)")
plt.ylabel("Yaw (rad)")
plt.title("Heading")

plt.tight_layout()
plt.show()
