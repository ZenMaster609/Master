# Per-cone perception output for trajectory/control consumers.
# Position is expressed in ConeDetectionArray.header.frame_id.

uint8 TRACK_STATE_TENTATIVE=0
uint8 TRACK_STATE_CONFIRMED=1
uint8 TRACK_STATE_STALE=2

string color
geometry_msgs/Point position
float32 confidence
uint32 track_id
uint8 track_state
float32 track_confidence
float32 color_confidence
uint32 seen_count
uint32 consecutive_seen_count
uint32 missed_count
builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen
