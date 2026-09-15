import psutil
try:
    p = psutil.Process(3492)
    p.kill()
    print("Killed 3492 successfully")
except Exception as e:
    print("3492 not running or error:", e)
