import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import os

# ============================================================
# CONSTANTS
# ============================================================

INDIVIDUAL_TABLE_HEADERS = ['place', 'video', 'athlete', 'grade', 'team', 'finish', 'point']
TEAM_TABLE_HEADERS       = ['place', 'tsTeam', 'point', 'wind', 'heat']

TIME_PATTERN = re.compile(
    r"\d+:\d{2}(?:\.\d+)?|\d+:\d+:\d{2}(?:\.\d+)?"
)  # mm:ss(.xx) or h:mm:ss(.xx)

TAG_AFTER_TIME = re.compile(r"^(PR|SR|NR|DNF|DNS|DQ|NT)$", re.IGNORECASE)


# ============================================================
# SHARED: extract_race_id
# ============================================================

def extract_race_id(url: str):
    match = re.search(r'results/(\d+)/', url)
    return match.group(1) if match else None


# ============================================================
# DETECTORS
# Each detector returns a score in [0, 1].
# We will pick the parser with the highest score.
# ============================================================

REQUIRED_HEADERS_KATIE = {"place", "video", "athlete", "grade", "team", "finish", "point"}
REQUIRED_HEADERS_COLE  = {"results", "print", "mile", "run"}  # loose hints
REQUIRED_HEADERS_MAX   = {"fr", "so", "jr", "sr"}             # class codes
REQUIRED_HEADERS_ADAM  = {"place", "athlete", "grade", "school", "time"}


def detect_katie(html: str) -> float:
    """
    Katie: table-based pages with td/ th classes like 'place', 'athlete', etc.
    """
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    # 1) Look for tables that have many of the expected classes
    tables = soup.find_all("table")
    if not tables:
        return 0.0

    best_hit = 0
    for tbl in tables:
        cell_classes = set()
        for cell in tbl.find_all(["td", "th"]):
            cls = cell.get("class", [])
            if isinstance(cls, str):
                cls = cls.split()
            for c in cls:
                cell_classes.add(c.strip().lower())
        hits = len(REQUIRED_HEADERS_KATIE.intersection(cell_classes))
        best_hit = max(best_hit, hits)

    if best_hit >= 3:
        score += 0.6
    elif best_hit >= 1:
        score += 0.3

    # 2) Presence of 'eventtable' style classes is a strong hint
    has_event_table = False
    for tbl in tables:
        cls = tbl.get("class", [])
        if isinstance(cls, str):
            cls = cls.split()
        tokens = [c.lower() for c in cls]
        if any("eventtable" in tok for tok in tokens):
            has_event_table = True
            break
    if has_event_table:
        score += 0.3

    # 3) Lots of links inside the table body (athlete/team URLs)
    links = soup.select("table tbody a[href]")
    if len(links) >= 5:
        score += 0.1

    return float(min(score, 1.0))


def detect_cole(html: str) -> float:
    """
    Cole: PRE-based results inside #meetResultsBody, no tables.
    Typical text looks like:
        'Boys 3 mile run 1. 10 Brandon Tavarez 23:25 PR Robert F. Kennedy ...'
    We key off:
      - presence of <pre> in meetResultsBody
      - repeated 'N.' place markers
      - repeated time-like tokens
    """
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    results_body = soup.find(id="meetResultsBody") or soup.find(class_="meetResultsBody")
    if not results_body:
        return 0.0

    pre_blocks   = results_body.find_all("pre")
    table_blocks = results_body.find_all("table")

    if not pre_blocks:
        return 0.0

    # flatten text
    text_all = " ".join(pre.get_text(" ", strip=True) for pre in pre_blocks)
    text_lower = text_all.lower()

    # 1) time tokens
    times = TIME_PATTERN.findall(text_all)
    if len(times) >= 8:
        score += 0.6
    elif len(times) >= 4:
        score += 0.4
    elif len(times) >= 1:
        score += 0.2

    # 2) place markers like "1." "2." etc
    place_markers = re.findall(r"\b\d+\.", text_all)
    if len(place_markers) >= 6:
        score += 0.3
    elif len(place_markers) >= 2:
        score += 0.2

    # 3) generic hints in header text
    if any(h in text_lower for h in REQUIRED_HEADERS_COLE):
        score += 0.1

    # penalize if tables exist (then it's probably not Cole)
    if table_blocks:
        score *= 0.3
    else:
        score += 0.1

    # penalize if heavy use of FR/SO/JR/SR (that smells like Max instead)
    grade_tokens = re.findall(r"\b(FR|SO|JR|SR)\b", text_all)
    if len(grade_tokens) >= 3:
        score *= 0.5

    return float(min(score, 1.0))


def detect_max(html: str) -> float:
    """
    Max: PRE-based results with FR/SO/JR/SR grade codes.
    """
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    results_body = soup.find(id="meetResultsBody") or soup.find(class_="meetResultsBody")
    if not results_body:
        return 0.0

    pre_blocks   = results_body.find_all("pre")
    table_blocks = results_body.find_all("table")
    if not pre_blocks:
        return 0.0

    text_all   = " ".join(pre.get_text(" ", strip=True) for pre in pre_blocks)
    text_lower = text_all.lower()

    # 1) FR/SO/JR/SR tokens
    grade_tokens = re.findall(r"\b(FR|SO|JR|SR)\b", text_all)
    if len(grade_tokens) >= 6:
        score += 0.6
    elif len(grade_tokens) >= 3:
        score += 0.4
    elif len(grade_tokens) >= 1:
        score += 0.2

    # 2) time tokens
    times = TIME_PATTERN.findall(text_all)
    if len(times) >= 5:
        score += 0.3
    elif len(times) >= 2:
        score += 0.2

    # 3) no tables -> more likely Max than Adam
    if not table_blocks:
        score += 0.1

    # mild hint words
    if "team scores" in text_lower:
        score += 0.1

    return float(min(score, 1.0))


def detect_adam(html: str) -> float:
    """
    Adam: table-based inside meetResultsBody with headers
    like Place / Athlete / Grade / School / Time.
    """
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    container = soup.find(id="meetResultsBody") or soup.find(class_="meetResultsBody")
    if not container:
        return 0.0

    tables = container.find_all("table")
    if not tables:
        return 0.0

    # 1) header coverage
    best_header_score = 0.0
    for tbl in tables:
        headers = {
            th.get_text(" ", strip=True).lower()
            for th in tbl.find_all(["th", "td"])
        }
        intersection = REQUIRED_HEADERS_ADAM.intersection(headers)
        if intersection:
            best_header_score = max(
                best_header_score,
                len(intersection) / len(REQUIRED_HEADERS_ADAM)
            )

    if best_header_score >= 0.6:
        score += 0.7
    elif best_header_score >= 0.3:
        score += 0.4

    # 2) presence of paging/filtering controls
    header_structure = soup.find("form", id="frmMeetResultsDetailFilter")
    if header_structure:
        select_box = header_structure.find("select", id="ddResultsPage")
        if select_box and select_box.find_all("option"):
            score += 0.2

    # 3) multiple tables usually means individual + team
    if len(tables) >= 2:
        score += 0.1

    return float(min(score, 1.0))


# ============================================================
# WRANGLERS
# ============================================================

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def wrangle_cole(html: str, race_url: str = None) -> pd.DataFrame:
    """
    Robust PRE parser for Cole-style pages (numeric grades).
    Handles both line-based and '1. 10 Name 23:25 PR Team ...' packed text.
    """
    soup = BeautifulSoup(html, "html.parser")
    results_div = soup.find("div", id="meetResultsBody") or soup.find("div", class_="meetResultsBody")
    if not results_div:
        return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)

    pre = results_div.find("pre")
    if not pre:
        return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)

    text = pre.get_text("\n", strip=True)

    # first try line-based parsing
    rows = []
    for raw_line in text.splitlines():
        line = _normalize_whitespace(raw_line)
        if not re.match(r"^\d+", line):
            continue

        # pattern: place [grade] name time [tag] team
        m = re.match(
            r"^(?P<place>\d+)\.?\s+"
            r"(?:(?P<grade>\d+)\s+)?"
            r"(?P<name>[A-Za-z',.\- ]+?)\s+"
            r"(?P<time>\d+:\d{2}(?:\.\d+)?|\d+:\d+:\d{2}(?:\.\d+)?)"
            r"(?:\s+(?P<tag>[A-Za-z]+))?\s+"
            r"(?P<team>[A-Za-z][A-Za-z .'\-]+)$",
            line
        )
        if not m:
            continue

        g = m.groupdict()
        finish = g["time"]
        # ignore tag (PR, SR, etc.) except we don't want to swallow time
        rows.append({
            "place": int(g["place"]),
            "video": None,
            "athlete": g["name"].strip(),
            "grade": int(g["grade"]) if g["grade"] is not None else pd.NA,
            "team": g["team"].strip(),
            "finish": finish,
            "point": pd.NA
        })

    # if we got enough rows, use them
    if len(rows) >= 3:
        return pd.DataFrame(rows, columns=INDIVIDUAL_TABLE_HEADERS)

    # otherwise, fall back to packed-text parsing:
    flat = _normalize_whitespace(text)

    packed_pattern = re.compile(
        r"(?P<place>\d+)\.\s+"
        r"(?:(?P<grade>\d+)\s+)?"
        r"(?P<name>[A-Za-z',.\- ]+?)\s+"
        r"(?P<time>\d+:\d{2}(?:\.\d+)?|\d+:\d+:\d{2}(?:\.\d+)?)"
        r"(?:\s+(?P<tag>[A-Za-z]+))?\s+"
        r"(?P<team>[A-Za-z][A-Za-z .'\-]+?)"
        r"(?=\s+\d+\.|$)"
    )

    rows = []
    for m in packed_pattern.finditer(flat):
        g = m.groupdict()
        finish = g["time"]
        rows.append({
            "place": int(g["place"]),
            "video": None,
            "athlete": g["name"].strip(),
            "grade": int(g["grade"]) if g["grade"] is not None else pd.NA,
            "team": g["team"].strip(),
            "finish": finish,
            "point": pd.NA
        })

    if not rows:
        return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)

    return pd.DataFrame(rows, columns=INDIVIDUAL_TABLE_HEADERS)


def wrangle_max(html: str, race_url: str = None):
    """
    PRE parser for Max-style pages with FR/SO/JR/SR grades.
    We keep your earlier pattern but with a bit of whitespace normalization.
    """
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", id="meetResultsBody") or soup.find("div", class_="meetResultsBody")
    if not container:
        return (
            pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS),
            pd.DataFrame(columns=TEAM_TABLE_HEADERS)
        )

    pre = container.find("pre")
    if not pre:
        return (
            pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS),
            pd.DataFrame(columns=TEAM_TABLE_HEADERS)
        )

    text = pre.get_text("\n", strip=True)
    text = _normalize_whitespace(text)

    sections = re.split(r'(?=\b[A-Z][A-Za-z/ &-]+ (?:Boys|Girls)\b)', text)

    rows = []

    line_pattern = re.compile(
        r'^(\d+)\s+([A-Za-z\'\-. ]+?)\s+(FR|SO|JR|SR)\s+'
        r'([A-Za-z\'\-. ]+?)\s+\d*:?[\d.]*\s+(\d+:\d+(?:\.\d+)?)\s+(\d+)?$'
    )

    for section in sections:
        section = section.strip()
        if not section:
            continue

        for raw_line in section.splitlines():
            line = _normalize_whitespace(raw_line)
            if not re.match(r'^\d+\s', line):
                continue

            m = line_pattern.match(line)
            if not m:
                continue

            place, athlete, grade, team, finish, point = m.groups()
            rows.append({
                "place": int(place),
                "video": None,
                "athlete": athlete.strip(),
                "grade": grade,
                "team": team.strip(),
                "finish": finish,
                "point": point if point else pd.NA
            })

    indiv_df = pd.DataFrame(rows, columns=INDIVIDUAL_TABLE_HEADERS)
    return indiv_df, pd.DataFrame(columns=TEAM_TABLE_HEADERS)


def wrangle_adam(html: str, race_url: str = None):
    """
    For now, Adam's wrangler simply returns empty; we rely on the
    robust table parser (Katie-style) for table pages.
    We keep this stub to preserve the groupmate structure.
    """
    return (
        pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS),
        pd.DataFrame(columns=TEAM_TABLE_HEADERS)
    )


def wrangle_katie(html: str, race_url: str = None):
    """
    Unused directly; robust table parser below plays Katie's role.
    """
    return (
        pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS),
        pd.DataFrame(columns=TEAM_TABLE_HEADERS)
    )


# ============================================================
# ROBUST TABLE PARSER (Katie-style but more tolerant)
# ============================================================

def extract_table_data(page_content: str, url: str):
    race_id = extract_race_id(url)
    soup    = BeautifulSoup(page_content, 'html.parser')
    tables  = soup.find_all('table')

    if not tables:
        print(f"   No tables found for URL: {url}")
        empty = {"individual": pd.DataFrame(), "team": pd.DataFrame()}
        meta  = pd.DataFrame([{
            "race_id": race_id,
            "url": url,
            "table_index": None,
            "table_type": "no_tables",
            "row_count": 0
        }])
        return empty, meta

    all_data = {"individual": [], "team": []}
    metadata = []

    indiv_headers_set = set(INDIVIDUAL_TABLE_HEADERS)
    team_headers_set  = set(TEAM_TABLE_HEADERS)

    for table_index, table in enumerate(tables, start=1):
        # Collect all classes in this table to decide type
        cell_classes = set()
        for cell in table.find_all(['td', 'th']):
            cls = cell.get('class', [])
            if isinstance(cls, str):
                cls = cls.split()
            for c in cls:
                cell_classes.add(c.strip())

        indiv_hits = indiv_headers_set.intersection(cell_classes)
        team_hits  = team_headers_set.intersection(cell_classes)

        if len(indiv_hits) >= 3 and len(indiv_hits) >= len(team_hits):
            table_type = "individual"
        elif len(team_hits) >= 3:
            table_type = "team"
        else:
            metadata.append({
                "race_id": race_id,
                "url": url,
                "table_index": table_index,
                "table_type": "unknown_headers",
                "row_count": 0
            })
            continue

        tbody = table.find("tbody")
        if tbody:
            rows = tbody.find_all("tr")
        else:
            rows = table.find_all("tr")

        added = 0
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue

            row_data = {
                "race_id": race_id,
                "race_url": url
            }

            for cell in cells:
                cls_list = cell.get("class", [])
                if isinstance(cls_list, str):
                    cls_list = cls_list.split()

                text_val = cell.get_text(" ", strip=True)

                for cls in cls_list:
                    cls = cls.strip()
                    if table_type == "individual" and cls in indiv_headers_set:
                        row_data[cls] = text_val
                        link = cell.find("a")
                        if link and link.get("href"):
                            row_data[f"{cls}_url"] = link.get("href")
                    elif table_type == "team" and cls in team_headers_set:
                        row_data[cls] = text_val
                        link = cell.find("a")
                        if link and link.get("href"):
                            row_data[f"{cls}_url"] = link.get("href")

            if table_type == "individual":
                place_str = str(row_data.get("place", "")).strip()
                if not place_str or not re.match(r"^\d+$", place_str):
                    continue
                if "athlete" not in row_data or "finish" not in row_data:
                    continue
                all_data["individual"].append(row_data)
                added += 1
            else:
                place_str = str(row_data.get("place", "")).strip()
                if not place_str or not re.match(r"^\d+$", place_str):
                    continue
                all_data["team"].append(row_data)
                added += 1

        metadata.append({
            "race_id": race_id,
            "url": url,
            "table_index": table_index,
            "table_type": table_type,
            "row_count": added
        })

    metadata_df = pd.DataFrame(metadata)
    indiv_df    = pd.DataFrame(all_data["individual"])
    team_df     = pd.DataFrame(all_data["team"])

    return {"individual": indiv_df, "team": team_df}, metadata_df


# ============================================================
# WRAPPED PARSER (detectors + wranglers + fallback)
# ============================================================

def extract_table_data_wrapped(page_content: str, url: str):
    race_id = extract_race_id(url)

    cole_score  = detect_cole(page_content)
    katie_score = detect_katie(page_content)
    max_score   = detect_max(page_content)
    adam_score  = detect_adam(page_content)

    scores = {
        "cole": cole_score,
        "katie": katie_score,
        "max": max_score,
        "adam": adam_score
    }

    best  = max(scores, key=scores.get)
    score = scores[best]

    print(f"   Detector scores: {scores}, best = {best} ({score:.2f})")

    try:
        if best == "cole" and score >= 0.70:
            print("   [OUR PARSER] Using COLE pre-parser")
            indiv_df = wrangle_cole(page_content, url)
            team_df  = pd.DataFrame(columns=TEAM_TABLE_HEADERS)
        elif best == "max" and score >= 0.70:
            print("   [OUR PARSER] Using MAX pre-parser")
            indiv_df, team_df = wrangle_max(page_content, url)
        elif best == "adam" and score >= 0.70:
            print("   [OUR PARSER] Using ADAM table parser (via robust fallback)")
            # Adam's wrangler is stub; rely on robust table parser
            data, meta = extract_table_data(page_content, url)
            meta["assigned_parser"] = "adam"
            return data, meta
        else:
            # Katie (or uncertain) -> robust table parser
            print("   [FALLBACK] Using robust table parser (Katie-style)")
            data, meta = extract_table_data(page_content, url)
            meta["assigned_parser"] = "katie_fallback"
            return data, meta

        meta = pd.DataFrame([{
            "race_id": race_id,
            "url": url,
            "assigned_parser": best,
            "table_index": None,
            "table_type": best,
            "row_count": len(indiv_df) + len(team_df),
            "detector_score": score
        }])

        return {"individual": indiv_df, "team": team_df}, meta

    except Exception as e:
        print(f"   ⚠ OUR WRANGLER ERROR ({best}) → falling back to robust table parser. Error: {e}")
        data, meta = extract_table_data(page_content, url)
        meta["assigned_parser"] = "katie_fallback_error"
        return data, meta


# ============================================================
# PROCESS URLS
# ============================================================

def process_urls_and_save_wrapped(urls):
    individual_results = pd.DataFrame()
    team_results       = pd.DataFrame()
    metadata_results   = pd.DataFrame()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        )
        page = browser.new_page()

        for url in urls:
            race_id = extract_race_id(url)
            print(f"\n🔍 Processing URL: {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=120000)
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_timeout(3000)

                try:
                    page.wait_for_selector("table", timeout=15000)
                except:
                    print("   ⚠ No table found after 15 seconds — continuing")

                html_content = page.content()

                data, metadata = extract_table_data_wrapped(html_content, url)

                if not data["individual"].empty:
                    individual_results = pd.concat(
                        [individual_results, data["individual"]],
                        ignore_index=True
                    )

                if not data["team"].empty:
                    team_results = pd.concat(
                        [team_results, data["team"]],
                        ignore_index=True
                    )

                if metadata is not None and not metadata.empty:
                    metadata_results = pd.concat(
                        [metadata_results, metadata],
                        ignore_index=True
                    )

            except Exception as e:
                print(f"   ERROR processing URL {url}: {e}")
                error_meta = pd.DataFrame([{
                    "race_id": race_id,
                    "url": url,
                    "assigned_parser": "error",
                    "table_index": '',
                    "table_type": f'error - {e}',
                    "row_count": 0,
                    "detector_score": None
                }])
                metadata_results = pd.concat(
                    [metadata_results, error_meta],
                    ignore_index=True
                )

        browser.close()

    return individual_results, team_results, metadata_results


# ============================================================
# DIAGNOSTIC MODE — SAMPLE SUBSET OF URLS
# ============================================================

if __name__ == "__main__":
    input_csv = r"C:\Users\coleg\OneDrive\Documents\Econ Research Lab\Kurtis-Econ-Research-Lab-Fall-2025\race_urls_2016.0.csv"

    df   = pd.read_csv(input_csv)
    # adjust n as you like; 80 is a decent compromise
    urls = df["race_url"].sample(n=80, random_state=42).tolist()

    print("\n==============================")
    print("  DIAGNOSTIC MODE: 80 URLs")
    print("==============================\n")

    individual, team, metadata = process_urls_and_save_wrapped(urls)

    # Ensure row_count numeric
    if "row_count" in metadata.columns:
        metadata["row_count"] = pd.to_numeric(metadata["row_count"], errors="coerce").fillna(0)
    else:
        metadata["row_count"] = 0

    print("\n=== PARSER FAILURE SUMMARY (SAMPLED 80 URLS) ===")
    if "assigned_parser" not in metadata.columns:
        metadata["assigned_parser"] = "unknown"

    summary = (
        metadata.groupby("assigned_parser")["row_count"]
        .agg(["count", lambda x: (x == 0).sum()])
        .rename(columns={"count": "urls_assigned", "<lambda_0>": "urls_with_zero_rows"})
    )
    summary["failure_rate"] = summary["urls_with_zero_rows"] / summary["urls_assigned"]
    print(summary)

    # Write diagnostic outputs
    output_dir = r"C:\Users\coleg\OneDrive\Documents\Econ Research Lab\Kurtis-Econ-Research-Lab-Fall-2025\output\diagnostic"
    os.makedirs(output_dir, exist_ok=True)
    individual.to_csv(os.path.join(output_dir, "diag_individual.csv"), index=False)
    team.to_csv(os.path.join(output_dir, "diag_team.csv"), index=False)
    metadata.to_csv(os.path.join(output_dir, "diag_metadata.csv"), index=False)

    print("\nDiagnostic complete. Files saved in 'output/diagnostic'.\n")

    # ========================================================
    # FULL RUN MODE (COMMENTED OUT FOR NOW)
    # ========================================================
    # If you want to run ALL URLs, comment out the block above
    # and uncomment this block:
    #
    # print("\n==============================")
    # print("  FULL RUN: ALL URLs")
    # print("==============================\n")
    #
    # urls_all = df["race_url"].tolist()
    # individual_all, team_all, metadata_all = process_urls_and_save_wrapped(urls_all)
    #
    # full_output_dir = r"C:\Users\coleg\OneDrive\Documents\Econ Research Lab\Kurtis-Econ-Research-Lab-Fall-2025\output\full_run"
    # os.makedirs(full_output_dir, exist_ok=True)
    # individual_all.to_csv(os.path.join(full_output_dir, "individual.csv"), index=False)
    # team_all.to_csv(os.path.join(full_output_dir, "team.csv"), index=False)
    # metadata_all.to_csv(os.path.join(full_output_dir, "metadata.csv"), index=False)
    #
    # print("\n✓ FULL RUN COMPLETE\n")
