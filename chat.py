#!/usr/bin/env python3
"""Talk to Alice.

    python3 chat.py                 # just the conversation
    python3 chat.py --show          # print the measurement behind every reply
    python3 chat.py --replay log.txt

The first thing you type becomes the frozen ground. Everything after is scored against it.
`--show` prints what she actually measured — the verdict, the echo, the tokens that arrived
and the ones that went quiet — so the reply can be checked against its own evidence rather
than taken on faith. That flag exists because a chatbot you cannot audit is a chatbot you
have to trust, and this one is trying not to ask for trust.

Type `:d 3` to tell her how far you feel from what you came for, 0–10. She takes that as
your word, not as a measurement, and says so.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from alice import Alice                                          # noqa: E402

DIM, BOLD, OFF = '\033[2m', '\033[1m', '\033[0m'
HUE = {'grounded': '\033[36m', 'advancing': '\033[32m', 'topic-drift': '\033[33m',
       'echo-spiral': '\033[35m', 'deliberation-stall': '\033[33m',
       'ungrammatical': '\033[31m'}


def render(out, show):
    if show:
        c = HUE.get(out['reason'], '')
        bits = [f"{c}{out['reason']}{OFF}", f"echo {out['echo']:.2f}"]
        if out['new']:
            bits.append(f"new ⟨{', '.join(out['new'][:4])}⟩")
        if out['dropped']:
            bits.append(f"dropped ⟨{', '.join(out['dropped'][:4])}⟩")
        print(f'{DIM}   [{"  ·  ".join(bits)}]{OFF}')
    print(f'{BOLD}alice{OFF}  {out["reply"]}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true', help='print the measurement behind each reply')
    ap.add_argument('--replay', help='feed a file of turns, one per line, and exit')
    a = ap.parse_args()

    alice, distance = Alice(), 5

    if a.replay:
        for line in pathlib.Path(a.replay).read_text().splitlines():
            if not line.strip():
                continue
            print(f'{DIM}you{OFF}    {line}')
            render(alice.hear(line, distance), a.show)
        return 0

    print(f'{DIM}alice · no model · powered by laserbrain{OFF}')
    print(f'{DIM}the first thing you say becomes the ground. :d N sets your distance, '
          f':q quits.{OFF}\n')
    while True:
        try:
            line = input(f'{DIM}you{OFF}    ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in (':q', ':quit', 'exit'):
            break
        if line.startswith(':d'):
            try:
                distance = max(0, min(10, int(line.split()[1])))
                print(f'{DIM}   [distance {distance} — your word, not a measurement]{OFF}\n')
            except (IndexError, ValueError):
                print(f'{DIM}   [usage: :d 0-10]{OFF}\n')
            continue
        if not line:
            continue
        render(alice.hear(line, distance), a.show)

    if alice.turns:
        print(f'{DIM}{alice.turns} turn(s). ground ⟨{", ".join(alice.ground)}⟩{OFF}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
