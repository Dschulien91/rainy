"""Language home pages and landing pages for search engines, one folder per language:

  /de/  app home    /de/regenradar/koeln/   /de/regenradar/bayern/   /de/regenradar/deutschland/
  /en/  app home    /en/rain-radar/cologne/ /en/rain-radar/bavaria/  /en/rain-radar/germany/

Each page is a copy of index.html (the full app) with: <html lang>, its own <title>, description, canonical URL,
hreflang links to the same page in every language (+ x-default = the bare domain, which picks the language),
window.LANG (UI language, see lang/*.js), window.ALT_URLS (where the language switcher goes), the map preset
window.PRESET = {lat, lon, zoom, name} (the app opens that place at that zoom, see init() in index.html) and its own
text in the right-hand column (between <!--SEO--> and <!--/SEO-->). Also writes sitemap.xml and robots.txt.
While the site is not public, every page carries noindex (from index.html; Caddy also sends X-Robots-Tag).

Add a language: an entry in LANGS (folder name, place-name translations, texts) + lang/<code>.js for the app UI
+ legal pages /<code>/impressum.html and /<code>/datenschutz.html + the Caddyfile language choice.

Places: tools/landing_places.json (cities with coordinates and population, made once by --fetch via our server's
anonymous geocoding proxy /api/geo/search). Re-run after every change to index.html or lang/*.js:

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


# ---------- per-language texts ----------
# names: German place name -> name in this language (missing = same as German)
EN_NAMES = {
    "München": "Munich", "Köln": "Cologne", "Nürnberg": "Nuremberg", "Hannover": "Hanover", "Frankfurt am Main": "Frankfurt",
    "Braunschweig": "Brunswick", "Bayern": "Bavaria", "Hessen": "Hesse", "Niedersachsen": "Lower Saxony",
    "Nordrhein-Westfalen": "North Rhine-Westphalia", "Rheinland-Pfalz": "Rhineland-Palatinate", "Sachsen": "Saxony",
    "Sachsen-Anhalt": "Saxony-Anhalt", "Thüringen": "Thuringia", "Mecklenburg-Vorpommern": "Mecklenburg-Western Pomerania",
    "Deutschland": "Germany", "Land Berlin": "Berlin", "Freie und Hansestadt Hamburg": "Hamburg",
    "Freie Hansestadt Bremen": "Bremen",
}


def de_city(c, n, st, pop, state_link, neigh):
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
    {f'<p>Mehr Regenradar für ganz {st}: {state_link}.</p>' if state_link else ''}
    <h2>Regenradar in der Nähe von {n}</h2>
    <ul class="links">{neigh}</ul>
  </article>"""


def en_city(c, n, st, pop, state_link, neigh):
    return f"""  <article class="seo">
    <h1>Rain radar {n} – rain in the next 2 hours</h1>
    <p>Is it about to rain in {n}? The rain radar shows live where it is raining around {n} right now and where the
      rain clouds will move in the next two hours. The map is centred on {n}{' (population ' + pop + ')' if pop else ''},
      and the timeline below shows to the minute when the rain starts and when it stops.</p>
    <h2>Live radar for {n} and the surrounding area</h2>
    <p>It is based on the radar images of the German Weather Service (DWD), updated every five minutes. From the
      direction the rain areas are moving, rainy forecasts {n} in 5-minute steps. Light drizzle that the radar
      cannot see is added from the DWD weather stations in {st}.</p>
    <h2>Weather in {n}: 24 hours and 3 days</h2>
    <p>Beyond the next two hours, rainy shows the DWD weather models for {n}: rainfall, temperature and cloud cover for
      the next 24 hours and three days.</p>
    {f'<p>Rain radar for all of {st}: {state_link}.</p>' if state_link else ''}
    <h2>Rain radar near {n}</h2>
    <ul class="links">{neigh}</ul>
  </article>"""


def de_state(n, lis):
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


def en_state(n, lis):
    return f"""  <article class="seo">
    <h1>Rain radar {n} – where is it raining right now?</h1>
    <p>The rain radar for {n} shows live where it is raining across the whole state and where the rain areas will move
      in the next two hours – based on the radar images of the German Weather Service (DWD), updated every five
      minutes.</p>
    <h2>Rain forecast for your place in {n}</h2>
    <p>Tap your place on the map or search for it: rainy then shows to the minute when the rain starts and stops where
      you are, plus temperature and rain for 24 hours and three days.</p>
    {'<h2>Cities in ' + n + '</h2><ul class="links">' + lis + '</ul>' if lis else ''}
  </article>"""


def de_country(st, ci):
    return f"""  <article class="seo">
    <h1>Regenradar Deutschland – live und mit 2-Stunden-Vorhersage</h1>
    <p>Wo regnet es gerade in Deutschland? Das Regenradar zeigt alle Regengebiete live auf der Karte und wohin sie in den
      nächsten zwei Stunden ziehen – aus den Radarbildern des Deutschen Wetterdienstes (DWD), alle fünf Minuten neu.</p>
    <h2>Regenradar nach Bundesland</h2>
    <ul class="links">{st}</ul>
    <h2>Regenradar für Städte</h2>
    <ul class="links">{ci}</ul>
  </article>"""


def en_country(st, ci):
    return f"""  <article class="seo">
    <h1>Rain radar Germany – live, with a 2-hour forecast</h1>
    <p>Where is it raining in Germany right now? The rain radar shows every rain area live on the map and where it will
      move in the next two hours – from the radar images of the German Weather Service (DWD), updated every five
      minutes.</p>
    <h2>Rain radar by state</h2>
    <ul class="links">{st}</ul>
    <h2>Rain radar for cities</h2>
    <ul class="links">{ci}</ul>
  </article>"""


EN_HOME_SEO = """  <article class="seo">
    <h1>Rain radar Germany – rain in the next 2 hours</h1>
    <p>rainy shows at a glance where it is raining in Germany right now and where the rain clouds will move in the next
      two hours. It is based on the radar images of the German Weather Service (DWD), updated every five minutes. For
      your location you see to the minute when the rain starts and when it stops.</p>
    <h2>How the rain radar works</h2>
    <p>The DWD weather radar scans the atmosphere across Germany and measures where precipitation falls and how heavy it
      is. From the movement in the latest radar images rainy calculates where the rain is heading – a forecast for the
      next two hours in 5-minute steps. Light drizzle that the radar cannot see is added from weather stations.</p>
    <h2>How accurate is the forecast?</h2>
    <p>For the next 15 to 30 minutes a radar forecast is very reliable. The further ahead it looks, the less certain it
      gets, because showers can form or dissolve. We continuously check our forecast against the rain that actually
      fell later.</p>
    <h2>24 hours and 3 days</h2>
    <p>Beyond two hours rainy shows the forecast of the DWD weather models: rain, temperature and cloud cover for the
      next 24 hours and three days.</p>
    <h2>Free, account optional</h2>
    <p>rainy works without an account. If you sign in (with Google or an email link), your favourites and settings
      follow you to all your devices.</p>
  </article>"""

LANGS = {
    "de": {
        "dir": "regenradar", "locale_pop": ".", "names": {}, "link": "Regenradar {name}",
        "title_city": "Regenradar {name} – Regen live & 2-Stunden-Vorhersage | rainy",
        "desc_city": "Regenradar {name}: Wo regnet es gerade? Live-Radar des DWD und minutengenaue Regenvorhersage für die "
                     "nächsten 2 Stunden in {name}.",
        "title_state": "Regenradar {name} – Regen live & 2-Stunden-Vorhersage | rainy",
        "desc_state": "Regenradar {name}: Live-Radar des DWD und Regenvorhersage für die nächsten 2 Stunden in ganz {name}.",
        "title_country": "Regenradar Deutschland – Regen live & 2-Stunden-Vorhersage | rainy",
        "desc_country": "Regenradar Deutschland: Wo regnet es gerade? Live-Radar des DWD und Regenvorhersage für die "
                        "nächsten 2 Stunden, für jeden Ort.",
        "title_home": "Regenradar Deutschland live – rainy",
        "desc_home": "Regenradar für Deutschland: Wo regnet es gerade? Live-Radar des DWD und minutengenaue "
                     "Regenvorhersage für die nächsten 2 Stunden.",
        "home_seo": None,  # the German text in index.html
        "city": de_city, "state": de_state, "country": de_country,
    },
    "en": {
        "dir": "rain-radar", "locale_pop": ",", "names": EN_NAMES, "link": "Rain radar {name}",
        "title_city": "Rain radar {name} – live rain & 2-hour forecast | rainy",
        "desc_city": "Rain radar {name}: where is it raining right now? Live DWD radar and a rain forecast to the minute "
                     "for the next 2 hours in {name}.",
        "title_state": "Rain radar {name} – live rain & 2-hour forecast | rainy",
        "desc_state": "Rain radar {name}: live DWD radar and a rain forecast for the next 2 hours across {name}.",
        "title_country": "Rain radar Germany – live rain & 2-hour forecast | rainy",
        "desc_country": "Rain radar Germany: where is it raining right now? Live DWD radar and a rain forecast for the next "
                        "2 hours, for every place.",
        "title_home": "Rain radar Germany live – rainy",
        "desc_home": "Rain radar for Germany: where is it raining right now? Live DWD radar and a rain forecast to the "
                     "minute for the next 2 hours.",
        "home_seo": EN_HOME_SEO,
        "city": en_city, "state": en_state, "country": en_country,
    },
}


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


class Site:
    """Page paths per language: key = the German name of a place (or "" for the home page)."""

    def __init__(self):
        self.paths = {code: {} for code in LANGS}

    def name(self, code, de_name):
        return LANGS[code]["names"].get(de_name, de_name)

    def path(self, code, de_name):
        if de_name == "":
            return f"/{code}/"
        return f"/{code}/{LANGS[code]['dir']}/{slug(self.name(code, de_name))}/"

    def link(self, code, de_name):
        label = LANGS[code]["link"].format(name=self.name(code, de_name))
        return f'<a href="{self.path(code, de_name)}">{html.escape(label)}</a>'


def page(template, site, code, de_name, *, title, desc, preset, seo):
    url = SITE + site.path(code, de_name)
    alts = {c: site.path(c, de_name) for c in LANGS}
    hreflang = "".join(f'<link rel="alternate" hreflang="{c}" href="{SITE}{p}">\n' for c, p in alts.items())
    head = (f'<base href="/">\n<title>{html.escape(title)}</title>\n'
            f'<meta name="description" content="{html.escape(desc)}">\n'
            f'<link rel="canonical" href="{url}">\n{hreflang}'
            f'<link rel="alternate" hreflang="x-default" href="{SITE}/">\n'
            f'<meta property="og:title" content="{html.escape(title)}">\n'
            f'<meta property="og:description" content="{html.escape(desc)}">\n'
            f'<meta property="og:url" content="{url}">\n'
            f'<script>window.LANG = {json.dumps(code)}; window.ALT_URLS = {json.dumps(alts)};'
            + (f' window.PRESET = {json.dumps(preset, ensure_ascii=False)};' if preset else '') + '</script>')
    out = template.replace("<title>rainy</title>", head, 1)
    out = re.sub(r'<html lang="[^"]*">', f'<html lang="{code}">', out, count=1)
    if seo:  # else the general (German) text of index.html stays
        out = re.sub(r"<!--SEO-->.*?<!--/SEO-->", lambda _: f"<!--SEO-->\n{seo}\n<!--/SEO-->", out, count=1, flags=re.S)
    return out


def main():
    if "--fetch" in sys.argv:
        return fetch()
    with open(os.path.join(ROOT, "index.html"), encoding="utf-8") as f:
        template = f.read()
    with open(PLACES, encoding="utf-8") as f:
        cities = json.load(f)
    site, urls = Site(), []

    def write(code, de_name, content):
        p = site.path(code, de_name)
        d = os.path.join(ROOT, *p.strip("/").split("/"))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "index.html"), "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        urls.append(SITE + p)

    by_pop = sorted(cities, key=lambda c: -c["population"])
    for code, L in LANGS.items():
        nm = lambda de_name: html.escape(site.name(code, de_name))
        write(code, "", page(template, site, code, "", title=L["title_home"], desc=L["desc_home"], preset=None,
                             seo=L["home_seo"]))
        for c in cities:
            near = sorted((o for o in cities if o is not c), key=lambda o: km(c, o))[:8]
            neigh = "".join(f"<li>{site.link(code, o['name'])} ({round(km(c, o))} km)</li>" for o in near)
            state = c["state"] if c["name"] not in CITY_STATES and any(s[0] == c["state"] for s in STATES) else None
            pop = f"{c['population']:,}".replace(",", L["locale_pop"]) if c["population"] else ""
            name = site.name(code, c["name"])
            write(code, c["name"], page(
                template, site, code, c["name"],
                title=L["title_city"].format(name=name), desc=L["desc_city"].format(name=name),
                preset={"lat": c["lat"], "lon": c["lon"], "zoom": zoom_for(c["population"]), "name": name},
                seo=L["city"](c, nm(c["name"]), nm(c["state"]), pop, site.link(code, state) if state else "", neigh)))
        for de_name, lat, lon, z in STATES:
            members = [c for c in by_pop if c["state"] == de_name]
            lis = "".join(f"<li>{site.link(code, c['name'])}</li>" for c in members)
            name = site.name(code, de_name)
            write(code, de_name, page(template, site, code, de_name,
                                      title=L["title_state"].format(name=name), desc=L["desc_state"].format(name=name),
                                      preset={"lat": lat, "lon": lon, "zoom": z, "name": name}, seo=L["state"](nm(de_name), lis)))
        de_name, lat, lon, z = COUNTRY
        st = "".join(f"<li>{site.link(code, s[0])}</li>" for s in STATES)
        ci = "".join(f"<li>{site.link(code, c['name'])}</li>" for c in by_pop[:30])
        write(code, de_name, page(template, site, code, de_name, title=L["title_country"], desc=L["desc_country"],
                                  preset={"lat": lat, "lon": lon, "zoom": z, "name": site.name(code, de_name)},
                                  seo=L["country"](st, ci)))
    with open(os.path.join(ROOT, "sitemap.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        f.writelines(f"  <url><loc>{u}</loc></url>\n" for u in urls)
        f.write("</urlset>\n")
    with open(os.path.join(ROOT, "robots.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /tiles/\nDisallow: /om/\nSitemap: {SITE}/sitemap.xml\n")
    print(len(urls), "pages in", len(LANGS), "languages, sitemap.xml, robots.txt")


if __name__ == "__main__":
    main()
