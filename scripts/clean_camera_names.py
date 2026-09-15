import sqlite3

conn = sqlite3.connect("output/sentinel.db")
c = conn.cursor()

# 1. Update CAM_08 to use the local high-speed MJPEG stream (CAM_08_0730.mp4)
c.execute("""
    UPDATE cameras 
    SET url = 'http://127.0.0.1:8091/stream.mjpg',
        stream_url = 'http://127.0.0.1:8091/stream.mjpg',
        name = 'Junagadh - Majewadi Gate'
    WHERE camera_id = 'CAM_08'
""")

# 2. Delete the duplicate CAM_33 / LIVE DEMO camera
c.execute("""
    DELETE FROM cameras 
    WHERE id = 'CAM_33' OR camera_id = 'LIVE DEMO - Majevadi Gate (CAM_08 feed)'
""")

# 3. Migrate all vehicle tracks to CAM_08
c.execute("""
    UPDATE vehicle_track 
    SET camera_id = 'CAM_08' 
    WHERE camera_id = 'LIVE DEMO - Majevadi Gate (CAM_08 feed)'
""")

conn.commit()
print("Successfully cleaned up cameras table and migrated tracks. Total changes:", conn.total_changes)

# Verify
c.execute("SELECT camera_id, name, stream_url FROM cameras WHERE camera_id = 'CAM_08'")
print("CAM_08 in DB:", c.fetchone())

c.execute("SELECT id, camera_id, plate_text, plate_confidence FROM vehicle_track WHERE plate_text IS NOT NULL ORDER BY id DESC LIMIT 5")
print("Latest 5 plates:")
for row in c.fetchall():
    print(" ", row)

conn.close()
