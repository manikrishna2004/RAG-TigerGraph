"""Deterministic structured layer over Olympic event infoboxes (no LLM, no GPU).

  python structured_events.py build [path/to/corpus.jsonl]   # parse all event pages, write events.json, print coverage
  python structured_events.py ask "question text"            # route one question, print evidence
  python structured_events.py runq questions.txt             # same for every question in a file
  python structured_events.py venue "Royal Artillery"        # diagnostic: every event page mentioning that venue

In code:
  from structured_events import structured_lookup
  ev = structured_lookup(question)     # evidence string, or None if the question is not one of the templates

Nothing is guessed: missing fields are reported as n/a, dates taken from the page body (not the infobox)
are labelled, and unmatched questions return a clear NOT FOUND.
"""
import datetime as dt
import json
import os
import re
import sys
import unicodedata
from datetime import date as _date
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE_DIR, "docs", "corpus.jsonl")
EVENTS_JSON = os.path.join(BASE_DIR, "events.json")

TITLE_RE = re.compile(r"^(?P<sport>.+?) at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics"
                      r"(?:\s*[\u2013\u2014-]\s*(?P<event>.+))?$")
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                      "august", "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
_MON = "|".join(sorted(MONTHS, key=len, reverse=True))
_DM = re.compile(rf"\b(\d{{1,2}})(?:\s*\u2013\s*(\d{{1,2}}))?\s+({_MON})\b(?:,?\s+(\d{{4}}))?", re.I)
_MD = re.compile(rf"\b({_MON})\s+(\d{{1,2}})(?:\s*\u2013\s*(\d{{1,2}}))?(?!\d)(?:,?\s+(\d{{4}}))?", re.I)


# ---------------------------------------------------------------- normalisation / parsing
def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("\u2019", "'").replace("'", "")
    s = s.replace("\u2019", "'").replace("'", "")
    s = s.replace("+", " plus ")          # keep "+80 kg" distinct from "80 kg"
    s = re.sub(r"\bkilomet(?:er|re)s?\b", "kilometre", s)
    s = re.sub(r"\bmet(?:er|re)s?\b", "metre", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def to_int(v):
    m = re.search(r"\d[\d,]*", v or "")
    return int(m.group(0).replace(",", "")) if m else None


def _date(y, mon, d):
    try:
        return dt.date(int(y), MONTHS[mon.lower()], int(d))
    except (KeyError, ValueError, TypeError):
        return None


def parse_dates(s, default_year=None):
    """Span (first_date, last_date) of every date mentioned, or None. Handles '8–14 August 2016', '3 to 4 August',
    '28 July – 4 August 2012', 'August 21–22', '12 and 14 February 2022', 'July 28, 2012 (heats)July 29, 2012 (final)'."""
    s = (s or "").replace("&", " and ")
    s = re.sub(r"\s*(?:\u2013|\u2014|-|\bto\b)\s*", "\u2013", s)
    s = re.sub(rf"\b(\d{{1,2}}) and (\d{{1,2}}) ({_MON})\b", r"\1 \3 and \2 \3", s, flags=re.I)
    ym = re.search(r"\b(1[89]\d\d|20\d\d)\b", s)
    fallback = ym.group(1) if ym else default_year   # a year written elsewhere in the string beats the default
    out = []
    for m in _DM.finditer(s):
        y = m.group(4) or fallback
        out += [_date(y, m.group(3), d) for d in (m.group(1), m.group(2)) if d]
    for m in _MD.finditer(s):
        y = m.group(4) or fallback
        out += [_date(y, m.group(1), d) for d in (m.group(2), m.group(3)) if d]
    out = [d for d in out if d]
    return (min(out), max(out)) if out else None

def parse_infobox(text):
    """-> (kind, fields, rest_of_page)"""
    m = re.match(r"\s*\[Infobox ([^\]]+)\]\s*\n", text or "")
    if not m:
        return None, {}, ""
    fields, pos = {}, m.end()
    for line in text[m.end():].split("\n"):
        pos += len(line) + 1
        if not line.strip():
            break
        k, sep, v = line.strip().partition(":")
        if sep:
            fields[k.strip()] = v.strip()
    return m.group(1).strip(), fields, text[pos:]


def first_sentence(rest):
    return re.split(r"(?<=[a-z0-9\)])\.\s", rest.strip(), maxsplit=1)[0][:400]


# ---------------------------------------------------------------- build
_MONTH = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12
}
_MON_PAT = '|'.join(_MONTH)

def extract_dates_from_intro(intro: str, year_hint: int = None):
    from datetime import date as _date
    t = intro.lower()
    ym = re.search(r'\b(19|20)\d{2}\b', t)
    year = int(ym.group()) if ym else year_hint
    if not year:
        return None, None

    # "from DD to DD Month"  e.g. "from 11 to 17 August"
    m = re.search(rf'from (\d{{1,2}}) to (\d{{1,2}}) ({_MON_PAT})(?:\s+(\d{{4}}))?', t)
    if m:
        d1, d2, mon = int(m.group(1)), int(m.group(2)), _MONTH[m.group(3)]
        y = int(m.group(4)) if m.group(4) else year
        return _date(y, mon, d1), _date(y, mon, d2)

    # "DD–DD Month"  e.g. "11–20 August 2016"
    m = re.search(rf'(\d{{1,2}})[–\-](\d{{1,2}}) ({_MON_PAT})(?:\s+(\d{{4}}))?', t)
    if m:
        d1, d2, mon = int(m.group(1)), int(m.group(2)), _MONTH[m.group(3)]
        y = int(m.group(4)) if m.group(4) else year
        return _date(y, mon, d1), _date(y, mon, d2)

    # "on DD and DD Month"  e.g. "on 3 and 4 August"
    m = re.search(rf'on (\d{{1,2}}) and (\d{{1,2}}) ({_MON_PAT})(?:\s+(\d{{4}}))?', t)
    if m:
        d1, d2, mon = int(m.group(1)), int(m.group(2)), _MONTH[m.group(3)]
        y = int(m.group(4)) if m.group(4) else year
        return _date(y, mon, d1), _date(y, mon, d2)

    # "on DD Month YYYY"  e.g. "on 28 July 2012"
    m = re.search(rf'on (\d{{1,2}}) ({_MON_PAT}) (\d{{4}})', t)
    if m:
        d, mon, y = int(m.group(1)), _MONTH[m.group(2)], int(m.group(3))
        return _date(y, mon, d), _date(y, mon, d)

    # "on DD Month" with year from context  e.g. "on 14 February"
    m = re.search(rf'on (\d{{1,2}}) ({_MON_PAT})\b', t)
    if m:
        d, mon = int(m.group(1)), _MONTH[m.group(2)]
        return _date(year, mon, d), _date(year, mon, d)

    return None, None

def build_events(corpus=CORPUS):
    events, stats, kinds, seen = [], Counter(), Counter(), set()
    bad_titles, bad_dates = [], []
    for line in open(corpus, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        stats["records"] += 1
        kind, f, rest = parse_infobox(d.get("text", ""))
        if kind:
            kinds[kind] += 1
        if not kind or "olympic event" not in kind.lower():
            continue
        stats["event_infoboxes"] += 1
        title = (d.get("title") or "").strip()
        m = TITLE_RE.match(title)
        if not m:
            bad_titles.append(title)
            continue
        if title in seen:
            stats["duplicate_titles"] += 1
            continue
        seen.add(title)
        year = int(m["year"])
        rng, src = parse_dates(f.get("dates", ""), year), "infobox"
        if f.get("dates") and not rng:
            bad_dates.append(f["dates"])
        if not rng:
            rng, src = parse_dates(first_sentence(rest), year), "body"
        if not rng:
            s, e = extract_dates_from_intro(first_sentence(rest), year_hint=year)
            if s:
                rng, src = (s, e), "intro"
        if not rng:
            src = None
        ev_title = (m["event"] or "").strip()
        events.append(dict(
            title=title, doc_id=d.get("doc_id"), sport=m["sport"].strip(), year=year, season=m["season"],
            event=ev_title or f.get("event", "").strip(), event_ib=f.get("event", "").strip(),
            venue=f.get("venue", ""), dates=f.get("dates", ""),
            start=rng[0].isoformat() if rng else None, end=rng[1].isoformat() if rng else None, dates_src=src,
            competitors=to_int(f.get("competitors")), nations=to_int(f.get("nations")),
            gold=f.get("gold", ""), gold_noc=f.get("goldNOC", ""), silver=f.get("silver", ""),
            silver_noc=f.get("silverNOC", ""), bronze=f.get("bronze", ""), bronze_noc=f.get("bronzeNOC", ""),
            prev=to_int(f.get("prev")), next=to_int(f.get("next"))))
    return events, dict(stats=dict(stats), kinds=kinds.most_common(10), bad_titles=bad_titles, bad_dates=bad_dates)


def coverage_report(events, info):
    n = len(events)
    print(f"records: {info['stats'].get('records', 0):,} | event infoboxes: {info['stats'].get('event_infoboxes', 0):,} "
          f"| parsed events: {n:,} | duplicate titles skipped: {info['stats'].get('duplicate_titles', 0)}")
    print("infobox kinds seen:", info["kinds"])
    print(f"title not parsed: {len(info['bad_titles'])}", info["bad_titles"][:5])
    print(f"infobox dates present but not parsed: {len(info['bad_dates'])}", info["bad_dates"][:8])
    src = Counter(e["dates_src"] for e in events)
    print(f"date span source: infobox {src.get('infobox', 0):,} | body text {src.get('body', 0):,} | none {src.get(None, 0):,}")
    for k in ("venue", "dates", "competitors", "nations", "gold", "silver", "bronze", "prev", "next"):
        have = sum(1 for e in events if e[k] not in (None, ""))
        print(f"  {k:12s} {have:6,}/{n:,}")
    by = Counter((e["year"], e["season"]) for e in events)
    print("games (year season: events):", ", ".join(f"{y} {s[:1]}:{c}" for (y, s), c in sorted(by.items())))


_EVENTS = None


def events():
    global _EVENTS
    if _EVENTS is None:
        if os.path.exists(EVENTS_JSON):
            _EVENTS = json.load(open(EVENTS_JSON, encoding="utf-8"))["events"]
        else:
            _EVENTS = build_events()[0]
    return _EVENTS


# ---------------------------------------------------------------- lookups
def _fmt(e):
    def p(name, noc):
        return f"{name} ({noc})" if name and noc else (name or "n/a")
    if e["dates_src"] == "infobox":
        dates = e["dates"]
    elif e["dates_src"] == "body":
        dates = f"{e['start']}..{e['end']} (inferred from page text, infobox has no usable dates)"
    else:
        dates = "n/a"
    return (f"{e['title']} | venue: {e['venue'] or 'n/a'} | dates: {dates} | "
            f"competitors: {e['competitors'] if e['competitors'] is not None else 'n/a'} | "
            f"nations: {e['nations'] if e['nations'] is not None else 'n/a'} | "
            f"gold: {p(e['gold'], e['gold_noc'])} | silver: {p(e['silver'], e['silver_noc'])} | "
            f"bronze: {p(e['bronze'], e['bronze_noc'])}")


def _games(year, season=None):
    return [e for e in events() if e["year"] == int(year) and (not season or e["season"].lower() == season.lower())]


def _sport(evs, sport):
    return [e for e in evs if norm(e["sport"]) == norm(sport)]


def _find_event(cands, event):
    q = norm(event)
    exact = [e for e in cands if q in (norm(e["event"]), norm(e.get("event_ib", "")))]
    if exact:
        return exact, True
    toks = set(q.split())
    return [e for e in cands if toks and toks <= set(norm(e["event"]).split())], False


def q_count_over(sport, year, season, n):
    evs = _sport(_games(year, season), sport)
    if not evs:
        return f"[structured] NOT FOUND: no event pages for {sport} at the {year} {season} Olympics."
    over = sorted([e for e in evs if e["competitors"] is not None and e["competitors"] > n],
                  key=lambda e: -e["competitors"])
    unk = [e for e in evs if e["competitors"] is None]
    out = [f"[structured] {sport} at the {year} {season} Olympics: {len(evs)} event pages, "
           f"{len(over)} with more than {n} competitors."]
    out += [f"  - {e['event']}: {e['competitors']} competitors" for e in over]
    if unk:
        out.append(f"  ({len(unk)} pages have no competitors field and are not counted: "
                   + "; ".join(e["event"] for e in unk[:10]) + ")")
    return "\n".join(out)


def q_max_competitors(sport, year, season):
    evs = [e for e in _sport(_games(year, season), sport) if e["competitors"] is not None]
    if not evs:
        return f"[structured] NOT FOUND: no {sport} events with a competitors field at the {year} {season} Olympics."
    top = max(e["competitors"] for e in evs)
    win = [e for e in evs if e["competitors"] == top]
    return (f"[structured] Highest competitors among {len(evs)} {sport} events at the {year} {season} Olympics: "
            f"{top} -> " + "; ".join(f"{e['event']} (page: {e['title']})" for e in win))


def q_nations(title):
    m = TITLE_RE.match(title.strip())
    if not m:
        return None
    hit, exact = _find_event(_sport(_games(m["year"], m["season"]), m["sport"]), m["event"] or "")
    if not hit:
        return f"[structured] NOT FOUND: {title}"
    return "[structured] " + ("" if exact else "(closest matches, not exact) ") + "\n".join(_fmt(e) for e in hit[:5])


def q_venue_date(venue, dates, year=None, season=None):
    rng = parse_dates(dates, year)
    if not rng:
        return f"[structured] NOT FOUND: could not parse date '{dates}'."
    qs, qe = rng
    nv = norm(venue)
    pool = _games(qs.year, season)

    def vmatch(e):
        ne = norm(e["venue"])
        return bool(nv and ne and (nv in ne or ne in nv))

    dated = [(e, (dt.date.fromisoformat(e["start"]), dt.date.fromisoformat(e["end"]))) for e in pool if e["start"]]
    exact = [e for e, s in dated if vmatch(e) and s == (qs, qe)]
    within = [e for e, s in dated if vmatch(e) and s[0] <= qs and qe <= s[1]]
    hit = exact or within
    if hit:
        note = "" if len(hit) == 1 else f" ({len(hit)} events match; the question may need disambiguation)"
        return "[structured] Events at that venue on those dates" + note + ":\n" + "\n".join("  " + _fmt(e) for e in hit[:12])
    venue_only = [e for e in pool if vmatch(e)]
    if venue_only:
        return ("[structured] No event at that venue has confirmed dates matching. Events at that venue in that Games "
                "(dates NOT confirmed, do not treat as an answer):\n" + "\n".join("  " + _fmt(e) for e in venue_only[:12]))
    same_day = [e for e, s in dated if s == (qs, qe)]
    if same_day:
        return ("[structured] NOT FOUND for that venue. Events with the exact same dates (venue not matched):\n"
                + "\n".join("  " + _fmt(e) for e in same_day[:10]))
    return f"[structured] NOT FOUND: no event with venue '{venue}' on {qs}..{qe}."


def previous_games(year, season):
    """Data-derived only: the 'prev' field of that year's pages, else the year whose pages point 'next' at it."""
    c = Counter(e["prev"] for e in _games(year, season) if e["prev"])
    if c:
        return c.most_common(1)[0][0]
    c = Counter(e["year"] for e in events() if e["season"].lower() == season.lower() and e["next"] == int(year))
    return c.most_common(1)[0][0] if c else None


def q_prev_games_event(phrase, season, year):
    sports = sorted({e["sport"] for e in events()}, key=len, reverse=True)
    np_ = norm(phrase)
    sport = next((s for s in sports if np_.endswith(" " + norm(s))), None)
    if not sport:
        return f"[structured] NOT FOUND: could not identify the sport in '{phrase}'."
    ev_norm = np_[: -len(norm(sport))].strip()
    prev = previous_games(year, season)
    if not prev:
        return f"[structured] NOT FOUND: no 'prev' field on {year} {season} pages and no earlier page whose 'next' is {year}."
    hit, exact = _find_event(_sport(_games(prev, season), sport), ev_norm)
    head = f"[structured] Games immediately before {year} {season} (from the pages' prev/next fields): {prev}."
    if not hit:
        return head + f" NOT FOUND: no {sport} event matching '{ev_norm}' at the {prev} {season} Olympics."
    return head + (" " if exact else " (closest matches, not exact) ") + "\n" + "\n".join("  " + _fmt(e) for e in hit[:5])


# ---------------------------------------------------------------- router
S = r"(?P<season>Summer|Winter)"
ROUTES = [
    (re.compile(rf"how many (?P<sport>.+?) events at the (?P<year>\d{{4}}) {S} Olympics\s*had more than (?P<n>\d+)\s*competitors", re.I),
     lambda m: q_count_over(m["sport"], int(m["year"]), m["season"], int(m["n"]))),
    (re.compile(rf"which (?P<sport>.+?) event at the (?P<year>\d{{4}}) {S} Olympics\s*had the highest number of competitors", re.I),
     lambda m: q_max_competitors(m["sport"], int(m["year"]), m["season"])),
    (re.compile(r"how many nations competed in (?P<title>.+?)\s*\?*$", re.I),
     lambda m: q_nations(m["title"])),
    (re.compile(rf"gold medal in the event held at (?P<venue>.+?) on\s*(?P<dates>.+?)(?: at the (?P<year>\d{{4}}) {S} Olympics)?\s*\?*$", re.I),
     lambda m: q_venue_date(m["venue"], m["dates"], int(m["year"]) if m["year"] else None, m["season"])),
    (re.compile(rf"gold medal in the (?P<phrase>.+?) event at the\s*{S}\s*Olympics held immediately before (?P<year>\d{{4}})", re.I),
     lambda m: q_prev_games_event(m["phrase"], m["season"], int(m["year"]))),
]


def structured_lookup(question):
    for rx, fn in ROUTES:
        m = rx.search(question.strip())
        if m:
            return fn(m)
    return None


def cmd_venue(text, corpus=CORPUS):
    """Every Olympic-event page whose infobox/intro mentions `text` (venue diagnostic)."""
    q, shown = norm(text), 0
    for line in open(corpus, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        kind, f, rest = parse_infobox(d.get("text", ""))
        if not kind or "olympic event" not in kind.lower():
            continue
        head = norm(d["text"][:1500])
        if q in head:
            shown += 1
            if shown <= 25:
                print(f"- {d['title']}\n    infobox venue: {f.get('venue', '(none)')} | infobox dates: {f.get('dates', '(none)')}"
                      f"\n    intro: {first_sentence(rest)[:200]}")
    print(f"\n{shown} event pages mention '{text}' in their infobox or intro")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        path = sys.argv[2] if len(sys.argv) > 2 else CORPUS
        evs, info = build_events(path)
        json.dump({"events": evs}, open(EVENTS_JSON, "w", encoding="utf-8"), ensure_ascii=False)
        coverage_report(evs, info)
        print(f"\nwrote {EVENTS_JSON}")
    elif cmd == "ask" and len(sys.argv) > 2:
        print(structured_lookup(" ".join(sys.argv[2:])) or "(no template matched this question)")
    elif cmd == "runq" and len(sys.argv) > 2:
        for q in (l.strip() for l in open(sys.argv[2], encoding="utf-8")):
            if q and not q.startswith("#"):
                print("\nQ:", q)
                print(structured_lookup(q) or "(no template matched)")
    elif cmd == "venue" and len(sys.argv) > 2:
        cmd_venue(" ".join(sys.argv[2:]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
