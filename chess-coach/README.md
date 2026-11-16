# Schach-Coach (Android)

Holt deine beendeten Partien von Chess.com, lässt **Stockfish 19** jeden Zug auf dem Handy prüfen und
**Ole, die Coach-Eule**, erklärt dir Zug für Zug, was gut war und was nicht.

## Wie die Partien in die App kommen

Die App zeichnet nicht den Bildschirm auf. Chess.com stellt jede beendete Partie über die öffentliche API bereit
(`api.chess.com/pub/player/<name>/games/...`), mit allen Zügen, der Uhrzeit nach jedem Zug, Gegner, Wertungszahlen,
Zeitkontrolle, Eröffnung und wie die Partie endete (Matt, Aufgabe, Zeit …). Dafür braucht es nur deinen
Benutzernamen, kein Passwort.

- **Beim Öffnen der App** werden neue Partien geladen (ein bis zwei kleine Anfragen, danach nichts mehr).
- **Optional im Hintergrund**: Alle ~4 Stunden schaut die App nach neuen Partien (nur mit Netz und nicht
  fast leerem Akku) und schickt eine Benachrichtigung. Analysiert wird erst, wenn du die App öffnest.
- **Teilen**: In der Chess.com-App nach der Partie „Teilen“ → PGN oder Link → Schach-Coach. Dann ist die
  Partie sofort da.
- **PGN einfügen** geht auch (z. B. Partien von anderen Seiten).

Gegenüber einer Bildschirmaufnahme spart das praktisch den ganzen Akku: Während du spielst, läuft gar nichts.

## Bewertung der Züge (−3 bis +3)

Grundlage ist, wie stark sich deine **Gewinnchance** durch den Zug verändert (Stockfish-Bewertung umgerechnet in
Prozent). Das ist fairer als reine Bauerneinheiten: +5 auf +3 ist egal, 0 auf −2 entscheidet die Partie.

| Wert | Klasse | Bedeutung |
|---|---|---|
| +3 | !! Brillant | bester Zug und ein echtes Figurenopfer |
| +2 | ! Stark | der einzige gute Zug, alles andere wäre deutlich schlechter |
| +1 | ★ Bester Zug / 👍 Exzellent | Stockfishs Zug oder praktisch gleich gut (≤ 3 % Verlust) |
| 0 | ✓ Gut / → Erzwungen | kleiner Verlust (≤ 7 %) bzw. einziger legaler Zug |
| −1 | ?! Ungenau | 7–14 % Gewinnchance verschenkt |
| −2 | ? Fehler | 14–25 % |
| −3 | ?? Patzer | mehr als 25 % – oder ein Zug, der ein schnelles Matt zulässt |

Dazu: Genauigkeit pro Spieler (wie bei Lichess aus dem Verlust je Zug berechnet), Bewertungsgraph,
Bedenkzeit pro Zug („nur 1 Sekunde überlegt“) und ein „Nächster Fehler“-Knopf.

## Installation

Die APK baut GitHub Actions bei jedem Push (Workflow `.github/workflows/chess-coach-apk.yml`):

1. Auf GitHub unter **Releases → „Schach-Coach (neuester Build)“** `schach-coach.apk` auf dem Handy herunterladen
   (oder unter Actions → letzter Lauf → Artifact `schach-coach-apk`).
2. Öffnen und „Installation aus dieser Quelle zulassen“ bestätigen.
3. Chess.com-Benutzernamen eingeben – fertig.

Updates einfach drüber installieren: alle Builds sind mit demselben Schlüssel signiert, die Partien bleiben erhalten.

## Akku

- Stockfish läuft nur während einer Analyse und wird nach 20 s Leerlauf beendet.
- Jede Stellung wird genau einmal analysiert, das Ergebnis wird gespeichert.
- Einstellungen → Analysetiefe: *Schnell* / *Normal* / *Gründlich*.
- Einstellungen → Neue Partien analysieren: *Sofort*, *Beim Laden* (nur am Ladekabel, Standard) oder *Beim Öffnen*.

## Aufbau

```
chess-coach/
  web/              die eigentliche App (HTML/CSS/JS, läuft auch im Browser)
    js/pgn.js         PGN-Leser (inkl. Uhrzeiten)
    js/games.js       Chess.com-API, Partie-Datensätze
    js/engine.js      Stockfish (WASM) im Web Worker über UCI
    js/analysis.js    Analyse, Gewinnchance, Zugklassen, Genauigkeit
    js/coach.js       Oles Kommentare
    js/board.js       Brett, Pfeile, Markierungen
    js/avatar.js      Ole (SVG, Stimmungen)
    js/app.js         Oberfläche, Analyse-Warteschlange, Sync
    vendor/           chess.js, Stockfish 19 lite (stockfish.js), Figuren (cburnett)
  android/          WebView-Hülle (Java): Assets, Chess.com-Proxy, Vorlesen, Teilen, Hintergrund-Check
```

Im Browser testen: `cd chess-coach/web && python3 -m http.server 8000`, dann `http://localhost:8000` öffnen.

## Lizenzen

Stockfish / stockfish.js: GPLv3 · chess.js: BSD-2 · Figuren „cburnett“: CC BY-SA 3.0 (via chessground).
Die App nutzt die öffentliche Chess.com-API und ist kein Produkt von Chess.com.
