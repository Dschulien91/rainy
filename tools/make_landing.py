"""Language home page and landing pages for search engines. Everything German lives under /de/: the app home page
/de/ and one page per city, state (Bundesland) and country, e.g. /de/regenradar/koeln/, /de/regenradar/bayern/,
/de/regenradar/deutschland/. While the site is not public, every page carries noindex (from index.html).

Each page is a copy of index.html (the full app) with its own <title>, description, canonical URL, the map preset
window.PRESET = {lat, lon, zoom, name} (the app opens that place at that zoom, see init() in index.html) and its own
text in the right-hand column (between <!--SEO--> and <!--/SEO-->), linking the state, nearby cities and the other
pages. Also writes sitemap.xml and robots.txt.

Places: tools/landing_places.json (cities with coordinates and population, made once by --fetch via our server's
anonymous geocoding proxy /api/geo/search). Re-run after every change to index.html:

  python tools/make_landing.py --fetch     (refresh city coordinates; needs the local server on :8771)
  python tools/make_landing.py             (write the pages)
"""
import html
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SITE = "https://hellorainy.com"
LANG = "de"
OUT = os.path.join(ROOT, LANG, "regenradar")
PLACES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landing_places.json")

CITIES = """Berlin;Hamburg;München;Köln;Frankfurt am Main;Stuttgart;Düsseldorf;Leipzig;Dortmund;Essen;Bremen;Dresden;
Hannover;Nürnberg;Duisburg;Bochum;Wuppertal;Bielefeld;Bonn;Münster;Mannheim;Karlsruhe;Augsburg;Wiesbaden;
Mönchengladbach;Gelsenkirchen;Aachen;Braunschweig;Kiel;Chemnitz;Halle (Saale);Magdeburg;Freiburg im Breisgau;Krefeld;
Mainz;Lübeck;Erfurt;Oberhausen;Rostock;Kassel;Hagen;Potsdam;Saarbrücken;Hamm;Ludwigshafen am Rhein;
Mülheim an der Ruhr;Oldenburg;Osnabrück;Leverkusen;Heidelberg;Solingen;Darmstadt;Herne;Neuss;Regensburg;Paderborn;
Ingolstadt;Offenbach am Main;Fürth;Würzburg;Ulm;Heilbronn;Pforzheim;Wolfsburg;Göttingen;Bottrop;Reutlingen;Koblenz;
Bremerhaven;Recklinghausen;Bergisch Gladbach;Erlangen;Jena;Remscheid;Trier;Salzgitter;Moers;Siegen;Hildesheim;
Cottbus;Gütersloh;Kaiserslautern;Schwerin;Witten"""

# states: centre and zoom so the whole state fits the map (city states have city pages instead)
STATES = [
    ("Baden-Württemberg", 48.66, 9.00, 7), ("Bayern", 48.95, 11.40, 7), ("Brandenburg", 52.40, 13.00, 7),
    ("Hessen", 50.60, 9.00, 7), ("Mecklenburg-Vorpommern", 53.80, 12.50, 7), ("Niedersachsen", 52.70, 9.20, 7),
    ("Nordrhein-Westfalen", 51.45, 7.60, 7), ("Rheinland-Pfalz", 49.90, 7.40, 7), ("Saarland", 49.38, 6.95, 9),
    ("Sachsen", 50.95, 13.30, 8), ("Sachsen-Anhalt", 51.95, 11.70, 7), ("Schleswig-Holstein", 54.20, 9.80, 7),
    ("Thüringen", 50.90, 11.00, 8),
]
CITY_STATES = {"Berlin", "Hamburg", "Bremen"}
COUNTRY = ("Deutschland", 51.16, 10.45, 6)


def slug(name):
    s = name.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def zoom_for(pop):
    """Same rule as cityZoom() in the app: bigger cities get a wider view."""
    return 10 if pop > 1e6 else 11 if pop > 2e5 else 12 if pop > 3e4 else 13


def km(a, b):
    dlat, dlon = math.radians(b["lat"] - a["lat"]), math.radians(b["lon"] - a["lon"])
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a["lat"])) * math.cos(math.radians(b["lat"])) * math.sin(dlon / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def fetch():
    out = []
    for name in [c.strip() for c in CITIES.replace("\n", "").split(";") if c.strip()]:
        res = []
        for q_name in (name, name.split(" an der ")[0].split(" am ")[0]):  # "Mülheim an der Ruhr" is listed as "Mülheim"
            q = urllib.parse.urlencode({"name": q_name, "count": 5, "language": "de", "format": "json", "countryCode": "DE"})
            d = json.load(urllib.request.urlopen(f"http://localhost:8771/api/geo/search?{q}", timeout=20))
            res = [r for r in d.get("results", []) if r.get("country_code") == "DE"]
            if res:
                break
        if not res:
            print("not found:", name)
            continue
        r = max(res, key=lambda r: r.get("population") or 0)
        out.append({"name": name, "lat": round(r["latitude"], 4), "lon": round(r["longitude"], 4),
                    "population": r.get("population") or 0, "state": r.get("admin1", "")})
    with open(PLACES, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(len(out), "cities ->", PLACES)


def link(name, sl):
    return f'<a href="/{LANG}/regenradar/{sl}/">Regenradar {html.escape(name)}</a>'


def page(template, *, url, title, desc, preset, seo):
    head = (f'<base href="/">\n<title>{html.escape(title)}</title>\n'
            f'<meta name="description" content="{html.escape(desc)}">\n'
            f'<link rel="canonical" href="{url}">\n'
            f'<link rel="alternate" hreflang="{LANG}" href="{url}">\n'
            f'<meta property="og:title" content="{html.escape(title)}">\n'
            f'<meta property="og:description" content="{html.escape(desc)}">\n'
            f'<meta property="og:url" content="{url}">'
            + (f'\n<script>window.PRESET = {json.dumps(preset, ensure_ascii=False)};</script>' if preset else ''))
    out = template.replace("<title>rainy</title>", head, 1)
    if seo:  # else the general text of index.html stays
        out = re.sub(r"<!--SEO-->.*?<!--/SEO-->", lambda _: f"<!--SEO-->\n{seo}\n<!--/SEO-->", out, count=1, flags=re.S)
    return out


def city_text(c, state_slug, near):
    n, st = html.escape(c["name"]), html.escape(c["state"])
    pop = f"{c['population']:,}".replace(",", ".") if c["population"] else ""
    neigh = "".join(f"<li>{link(o['name'], slug(o['name']))} ({round(km(c, o))} km)</li>" for o in near)
    state_link = (f'<p>Mehr Regenradar für ganz {st}: {link(c["state"], state_slug)}.</p>'
                  if state_slug else "")
    return f"""  <article class="seo">
    <h1>Regenradar {n} – Regen in den nächsten 2 Stunden</h1>
    <p>Regnet es gleich in {n}? Das Regenradar zeigt dir live, wo es rund um {n} gerade regnet und wohin die
      Regenwolken in den nächsten zwei Stunden ziehen. Die Karte ist direkt auf {n}{' (' + pop + ' Einwohner)' if pop else ''}
      eingestellt, die Zeitleiste unten zeigt minutengenau, wann der Regen beginnt und wann er wieder aufhört.</p>
    <h2>Live-Radar für {n} und Umgebung</h2>
    <p>Grundlage sind die Radarbilder des Deutschen Wetterdienstes (DWD), die alle fünf Minuten neu erscheinen. Aus
      der Zugrichtung der Regengebiete entsteht eine Vorhersage für {n} in 5-Minuten-Schritten. Feiner Niesel, den
      das Radar nicht erfasst, wird aus den Messstationen des DWD in {st} ergänzt.</p>
    <h2>Wetter in {n}: 24 Stunden und 3 Tage</h2>
    <p>Über die nächsten zwei Stunden hinaus zeigt rainy für {n} die Wettermodelle des DWD: Regenmenge,
      Temperatur und Bewölkung für die nächsten 24 Stunden und drei Tage.</p>
    {state_link}
    <h2>Regenradar in der Nähe von {n}</h2>
    <ul class="links">{neigh}</ul>
  </article>"""


def state_text(name, cities):
    n = html.escape(name)
    lis = "".join(f"<li>{link(c['name'], slug(c['name']))}</li>" for c in cities)
    return f"""  <article class="seo">
    <h1>Regenradar {n} – wo regnet es gerade?</h1>
    <p>Das Regenradar für {n} zeigt live, wo es im ganzen Bundesland gerade regnet und wohin die Regengebiete in den
      nächsten zwei Stunden ziehen – auf Basis der Radarbilder des Deutschen Wetterdienstes (DWD), alle fünf Minuten
      aktualisiert.</p>
    <h2>Regenvorhersage für deinen Ort in {n}</h2>
    <p>Tippe auf der Karte auf deinen Ort oder suche ihn: rainy zeigt dir dann minutengenau, wann der Regen bei dir
      beginnt und endet, dazu Temperatur und Regen für 24 Stunden und drei Tage.</p>
    {'<h2>Städte in ' + n + '</h2><ul class="links">' + lis + '</ul>' if lis else ''}
  </article>"""


def country_text(states, cities):
    st = "".join(f"<li>{link(s[0], slug(s[0]))}</li>" for s in states)
    ci = "".join(f"<li>{link(c['name'], slug(c['name']))}</li>" for c in cities[:30])
    return f"""  <article class="seo">
    <h1>Regenradar Deutschland – live und mit 2-Stunden-Vorhersage</h1>
    <p>Wo regnet es gerade in Deutschland? Das Regenradar zeigt alle Regengebiete live auf der Karte und wohin sie in den
      nächsten zwei Stunden ziehen – aus den Radarbildern des Deutschen Wetterdienstes (DWD), alle fünf Minuten neu.</p>
    <h2>Regenradar nach Bundesland</h2>
    <ul class="links">{st}</ul>
    <h2>Regenradar für Städte</h2>
    <ul class="links">{ci}</ul>
  </article>"""


def main():
    if "--fetch" in sys.argv:
        return fetch()
    with open(os.path.join(ROOT, "index.html"), encoding="utf-8") as f:
        template = f.read()
    with open(PLACES, encoding="utf-8") as f:
        cities = json.load(f)
    urls = [f"{SITE}/{LANG}/"]
    # the language home page: the plain app with its general text, at /de/
    with open(os.path.join(ROOT, LANG, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page(template, url=f"{SITE}/{LANG}/", title="Regenradar Deutschland live – rainy",
                     desc="Regenradar für Deutschland: Wo regnet es gerade? Live-Radar des DWD und minutengenaue "
                          "Regenvorhersage für die nächsten 2 Stunden.", preset=None, seo=None))

    def write(sl, content):
        d = os.path.join(OUT, sl)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "index.html"), "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        urls.append(f"{SITE}/{LANG}/regenradar/{sl}/")

    state_slugs = {s[0]: slug(s[0]) for s in STATES}
    for c in cities:
        near = sorted((o for o in cities if o is not c), key=lambda o: km(c, o))[:8]
        sl = slug(c["name"])
        write(sl, page(template, url=f"{SITE}/{LANG}/regenradar/{sl}/",
                       title=f"Regenradar {c['name']} – Regen live & 2-Stunden-Vorhersage | rainy",
                       desc=f"Regenradar {c['name']}: Wo regnet es gerade? Live-Radar des DWD und minutengenaue "
                            f"Regenvorhersage für die nächsten 2 Stunden in {c['name']}.",
                       preset={"lat": c["lat"], "lon": c["lon"], "zoom": zoom_for(c["population"]), "name": c["name"]},
                       seo=city_text(c, None if c["name"] in CITY_STATES else state_slugs.get(c["state"]), near)))
    for name, lat, lon, z in STATES:
        members = sorted((c for c in cities if c["state"] == name), key=lambda c: -c["population"])
        write(slug(name), page(template, url=f"{SITE}/{LANG}/regenradar/{slug(name)}/",
                               title=f"Regenradar {name} – Regen live & 2-Stunden-Vorhersage | rainy",
                               desc=f"Regenradar {name}: Live-Radar des DWD und Regenvorhersage für die nächsten "
                                    f"2 Stunden in ganz {name}.",
                               preset={"lat": lat, "lon": lon, "zoom": z, "name": name}, seo=state_text(name, members)))
    name, lat, lon, z = COUNTRY
    write(slug(name), page(template, url=f"{SITE}/{LANG}/regenradar/{slug(name)}/",
                           title="Regenradar Deutschland – Regen live & 2-Stunden-Vorhersage | rainy",
                           desc="Regenradar Deutschland: Wo regnet es gerade? Live-Radar des DWD und Regenvorhersage "
                                "für die nächsten 2 Stunden, für jeden Ort.",
                           preset={"lat": lat, "lon": lon, "zoom": z, "name": name},
                           seo=country_text(STATES, sorted(cities, key=lambda c: -c["population"]))))
    with open(os.path.join(ROOT, "sitemap.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        f.writelines(f"  <url><loc>{u}</loc></url>\n" for u in urls)
        f.write("</urlset>\n")
    with open(os.path.join(ROOT, "robots.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /tiles/\nDisallow: /om/\nSitemap: {SITE}/sitemap.xml\n")
    print(len(urls), "pages, sitemap.xml, robots.txt")


if __name__ == "__main__":
    main()
