import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

PASSWORD = "6KL6-3ZBX-UJMA"
BASE     = "https://cctv.corp8.cloud"

options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

try:
    print("Logging in...")
    driver.get(BASE)
    time.sleep(3)
    driver.find_element(By.TAG_NAME, "input").send_keys(PASSWORD)
    driver.find_element(By.TAG_NAME, "button").click()
    time.sleep(5)
    print("Logged in:", driver.title)
    print()
    print("="*60)
    print("BROWSER WINDOW MEIN KARO:")
    print("1. Kisi bhi camera par click karo")
    print("2. Stream play hone do")
    print("3. Yahan 30 seconds wait karo")
    print("="*60)
    print()

    found_urls = set()
    t0 = time.time()

    while time.time() - t0 < 30:
        logs = driver.get_log("performance")
        for log in logs:
            try:
                msg = json.loads(log["message"])["message"]
                if msg["method"] in (
                    "Network.requestWillBeSent",
                    "Network.responseReceived",
                ):
                    params = msg["params"]
                    url = (
                        params.get("request", {}).get("url", "") or
                        params.get("response", {}).get("url", "")
                    )
                    if url and any(x in url for x in [
                        ".m3u8", ".ts", "stream", "live",
                        "hls", "chunk", "media", "segment"
                    ]):
                        if url not in found_urls:
                            found_urls.add(url)
                            print("URL:", url[:100])
            except Exception:
                pass
        time.sleep(1)

    print()
    print("="*60)
    print("CAPTURED URLs:")
    for url in sorted(found_urls):
        print(" ", url)

    with open("captured_stream_urls.txt", "w") as f:
        f.write("\n".join(sorted(found_urls)))
    print()
    print("Saved: captured_stream_urls.txt")

    # JS se video source nikalo
    print()
    print("Checking video elements...")
    sources = driver.execute_script("""
        var results = [];
        document.querySelectorAll('video').forEach(function(v) {
            results.push({
                src: v.src,
                currentSrc: v.currentSrc,
                paused: v.paused
            });
        });
        return results;
    """)
    print("Video sources:", sources)

    driver.save_screenshot("dashboard_capture.png")
    print("Screenshot: dashboard_capture.png")

finally:
    driver.quit()
    print("Done")