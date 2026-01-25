from __future__ import annotations

import argparse
import json
from pathlib import Path

OPENS = "([{（「『"
CLOSES = ")]}）」」』"
PAIR = {c: o for c, o in zip(CLOSES, OPENS)}

def check(text: str):
    st: list[tuple[str, int]] = []
    bad = None
    for i, ch in enumerate(text):
        if ch in OPENS:
            st.append((ch, i))
        elif ch in CLOSES:
            need = PAIR.get(ch)
            if (not st) or (st[-1][0] != need):
                bad = ("CLOSE", i, ch, st[-1] if st else None)
                break
            st.pop()
    leftover = st[-1] if st else None
    return bad, leftover

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture_json", help="fixture json path (contains pages[].text)")
    ap.add_argument("--page", type=int, default=1, help="page_number to inspect")
    ap.add_argument("--window", type=int, default=140, help="context window chars")
    args = ap.parse_args()

    d = json.loads(Path(args.fixture_json).read_text(encoding="utf-8"))
    pages = d.get("pages", [])
    page = next((p for p in pages if int(p.get("page_number", 0)) == args.page), None)
    if not page:
        raise SystemExit(f"page not found: {args.page}")

    t = page.get("text", "")
    bad, leftover = check(t)

    print("bad=", bad)
    print("leftover_open=", leftover)

    idx = None
    if bad:
        idx = bad[1]
    elif leftover:
        idx = leftover[1]

    if idx is not None:
        w = args.window
        print("\n---CONTEXT---")
        print(t[max(0, idx - w): idx + w])
        print("---AT---", idx, repr(t[idx:idx+30]))

if __name__ == "__main__":
    main()
