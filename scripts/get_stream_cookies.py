import sys
import time
import json
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

PASSWORD = "6KL6-3ZBX-UJMA"
BASE     = "https://cctv.corp8.cloud"

print("Opening browser...")

options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_experimental_option("excludeSwitches", ["enable-logging"])

driver = webdriver.Chrome(options=options)

wait = WebDriverWait(driver, 15)

try:
    print("Loading:", BASE)
    driver.get(BASE)
    time.sleep(3)
    print("Title:", driver.title)

    # Input field mein password daalo
    inp = wait.until(EC.presence_of_element_located((By.TAG_NAME, "input")))
    inp.clear()
    inp.send_keys(PASSWORD)
    print("Password entered")
    time.sleep(1)

    # Button click karo — FRESH element lo
    btn = wait.until(EC.element_to_be_clickable((By.TAG_NAME, "button")))
    btn_text = btn.text
    print("Clicking button:", btn_text[:30])
    btn.click()

    print("Waiting for page change...")
    time.sleep(6)

    print("After login URL:", driver.current_url)
    print("After login title:", driver.title)

    # Screenshot
    driver.save_screenshot("after_login.png")
    print("Screenshot saved: after_login.png")

    # Cookies lo
    cookies     = driver.get_cookies()
    cookie_dict = {c["name"]: c["value"] for c in cookies}

    print("\nCookies (%d):" % len(cookies))
    for name, val in cookie_dict.items():
        print("  %s = %s..." % (name, val[:30]))

    # Stream test karo
    print("\nTesting stream with cookies...")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    for k, v in cookie_dict.items():
        session.cookies.set(k, v)

    for cam in ["cam01", "cam08", "cam09"]:
        r = session.get(BASE + "/%s/index.m3u8" % cam, timeout=10)
        print("%s: HTTP %d | %s" % (
            cam, r.status_code,
            "M3U8 OK" if "#EXTM3U" in r.text else r.text[:80]
        ))

    # Save cookies
    with open("stream_cookies.json", "w") as f:
        json.dump(cookie_dict, f, indent=2)
    print("\nCookies saved: stream_cookies.json")

    # Agar stream accessible hai
    r = session.get(BASE + "/cam01/index.m3u8", timeout=10)
    if "#EXTM3U" in r.text:
        print("\n" + "="*50)
        print("SUCCESS — STREAMS ACCESSIBLE!")
        print("="*50)
        print("M3U8 content:")
        print(r.text[:400])
    else:
        print("\nStream still blocked after login")
        print("Current page might need more time or different URL")
        print("Check after_login.png screenshot")

        # LocalStorage aur sessionStorage bhi check karo
        local_storage = driver.execute_script(
            "return Object.fromEntries(Object.entries(localStorage))"
        )
        print("\nLocalStorage:")
        for k, v in local_storage.items():
            print("  %s = %s" % (k, str(v)[:50]))

        session_storage = driver.execute_script(
            "return Object.fromEntries(Object.entries(sessionStorage))"
        )
        print("\nSessionStorage:")
        for k, v in session_storage.items():
            print("  %s = %s" % (k, str(v)[:50]))

        # Save all tokens
        all_tokens = {**cookie_dict, **local_storage, **session_storage}
        with open("stream_tokens.json", "w") as f:
            json.dump(all_tokens, f, indent=2)
        print("\nAll tokens saved: stream_tokens.json")

finally:
    driver.quit()
    print("Browser closed")