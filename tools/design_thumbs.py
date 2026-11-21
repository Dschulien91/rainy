"""Sample pictures for the settings' design page (app v1.37): one small picture per map design and mode.

Opens the live app in headless Chromium (Playwright) with each design, hides everything but the map and saves
img/design/<design>-<mode>[-dark|-light].webp (360 x 240, about 10-30 KB each). "plain" follows the app colours, so it
gets a dark and a light picture. Sunny pictures should be made by daylight in Europe (the sun look fades at night).
Mode "base" is the bare map (rainy without the weather layer): the app shows it while a sunny picture is missing.

Usage:  python tools/design_thumbs.py [--modes rain,sun,base] [--designs bright,night] [--url https://hellorainy.com]
"""
import argparse
import io
import json
import os

from PIL import Image
from playwright.sync_api import sync_playwright

DESIGNS = ["plain", "liberty", "bright", "fiord", "earth", "gray", "vintage", "night"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "img", "design")
VIEW = "lat=50.6&lon=9.4&zoom=6"  # central Germany with its neighbours
HIDE = """body > *:not(#map) { visibility: hidden !important; }
  .leaflet-control-container, .leaflet-marker-pane, .leaflet-labels-pane svg, .leaflet-overlay-pane svg { display: none !important; }"""


def shot(browser, url, design, mode, theme):
    ctx = browser.new_context(viewport={"width": 540, "height": 900}, device_scale_factor=1, locale="de-DE",
                              user_agent="Mozilla/5.0")
    style = {"design": design, "designPick": True, "designSun": design, "theme": theme}
    ctx.add_init_script(f"""try {{ localStorage.setItem("rainy.style", {json.dumps(json.dumps(style))});
      sessionStorage.setItem("rainy.reloadMode", "{'sun' if mode == 'sun' else 'rain'}"); }} catch (e) {{}}""")
    page = ctx.new_page()
    page.goto(f"{url}/?{VIEW}", wait_until="domcontentloaded")
    page.add_style_tag(content=HIDE + (" .leaflet-weather-pane { display: none !important; }" if mode == "base" else ""))
    page.wait_for_timeout(9000)  # map, radar / clouds
    # the map's visible part ends above the bottom sheet: a 3:2 cut from just below the top bar
    img = Image.open(io.BytesIO(page.screenshot(clip={"x": 0, "y": 70, "width": 540, "height": 360})))
    img = img.convert("RGB").resize((360, 240), Image.LANCZOS)
    ctx.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="rain,sun")
    ap.add_argument("--url", default="https://hellorainy.com")
    ap.add_argument("--designs", default=",".join(DESIGNS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for mode in a.modes.split(","):
            for d in a.designs.split(","):
                for theme in (["dark", "light"] if d == "plain" else ["dark"]):
                    name = f"{d}-{mode}" + (f"-{theme}" if d == "plain" else "") + ".webp"
                    shot(browser, a.url, d, mode, theme).save(os.path.join(OUT, name), "WEBP", quality=72, method=6)
                    print(name, os.path.getsize(os.path.join(OUT, name)) // 1024, "KB")
        browser.close()


if __name__ == "__main__":
    main()
