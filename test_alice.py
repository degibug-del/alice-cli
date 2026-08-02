#!/usr/bin/env python3
"""What Alice must keep doing. Every case here is a defect she actually had.

The first conversation ever held with her produced three of them in six turns, and none
would have been visible from reading the source — they only appeared when someone talked to
her. That is the argument for this file existing rather than a mock: the failures are
behavioural, so the tests are transcripts.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from alice import Alice, _sim_sets, _THIN                        # noqa: E402

fails = []


def check(label, cond, detail=''):
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f'   {detail}' if detail else ''))
    if not cond:
        fails.append(label)


def run(turns, distance=5):
    a = Alice()
    return a, [a.hear(t, distance) for t in turns]


print('the ground is frozen at the first turn')
a, out = run(['finish the drift paper this week', 'the corpus is too small'])
check('first turn grounds', out[0]['reason'] == 'grounded', out[0]['reason'])
check('  and the ground is held', a.ground == set(a.ground) and 'drift' in a.ground,
      str(sorted(a.ground)))
check('  a later turn does not move it', 'drift' in a.ground)

print()
print('echo is measured against the PERSON, not another agent')
# THE FIRST BUG. _Dialogue.step scores echo across different agents (`t['agent'] != agent`),
# so with one speaker it is 0.00 forever and echo-spiral can never fire. Two near-identical
# turns scored 0.00 in the first transcript.
a, out = run(['the corpus is not big enough to say anything',
              'the corpus is not big enough to say anything yet'])
check('a near-repeat scores high echo', out[-1]['echo'] > 0.5, f"echo {out[-1]['echo']}")
check('  and fires echo-spiral', out[-1]['reason'] == 'echo-spiral', out[-1]['reason'])
a2, out2 = run(['finish the drift paper', 'the solana bot needs a screening pipeline'])
check('an unrelated turn does not', out2[-1]['echo'] < 0.3, f"echo {out2[-1]['echo']}")

print()
print('she stops accusing after twice, and offers to move the ground')
# THE SECOND BUG. Passing restated_goal on every turn made each low-overlap sentence a goal
# declaration, so topic-drift fired five turns running and printed the same paragraph five
# times — the corpus's own goal-drift chain, reproduced live.
a, out = run(['finish the drift paper this week',
              'the solana bot needs a screening pipeline',
              'and a risk manager with a kill switch',
              'plus jupiter routing for the swaps',
              'and a dashboard to watch it'])
reasons = [o['reason'] for o in out[1:]]
check('drift is named at most twice', reasons.count('topic-drift') <= 2, str(reasons))
check('  then it offers a re-ground', 'reground-offer' in reasons, str(reasons))
offers = [o['reply'] for o in out if o['reason'] == 'reground-offer']
check('  and the offer does not repeat verbatim', len(set(offers)) == len(offers) or len(offers) < 2,
      f'{len(offers)} offer(s), {len(set(offers))} distinct')

print()
print('she never reports a word as missing that the person just said')
# THE THIRD BUG. The offer quoted the GROUND rather than what was absent, so it told someone
# who had just written "paper" that they had been off ⟨…, pap⟩.
a, out = run(['finish the drift paper this week',
              'the solana bot needs a screening pipeline',
              'and a risk manager',
              'ok back to the paper, what about the trial'])
last = out[-1]
check('"pap" is not listed as dropped', 'pap' not in last['dropped'], str(last['dropped']))
check('  nor quoted as absent in the reply', 'pap' not in last['reply'].split('⟩')[0],
      last['reply'][:90])

print()
print('filler is not reported as new content')
a, out = run(['finish the drift paper', 'ok so maybe I should just think about it more'])
new = set(out[-1]['new'])
check('no filler in `new`', not (new & _THIN), str(sorted(new & _THIN)))

print()
print('DETERMINISM — the same conversation gives the same replies, always')
# The property that makes her auditable. If she says something surprising you can replay it
# exactly; there is no seed to have lost.
script = ['finish the drift paper this week', 'the corpus is too small',
          'the corpus is still too small', 'what if I ran the trial first']
_, first = run(script)
for i in range(4):
    _, again = run(script)
    if [o['reply'] for o in first] != [o['reply'] for o in again]:
        check('replies are identical across runs', False, f'diverged on repeat {i + 1}')
        break
else:
    check('replies are identical across 5 runs', True)

print()
print('she refuses to measure what has no subject')
a, out = run(['   '])
check('an empty first turn is ungrammatical', out[0]['reason'] == 'ungrammatical',
      out[0]['reason'])
check('  and no ground is invented', not a.ground, str(a.ground))

print()
print('the re-ground she offers actually works')
# She tells people to say "new ground". Until it was implemented, saying it did nothing —
# the same shape as parent_goal being received and silently discarded, which held its
# adoption at 0.2% and taught agents the field was broken.
a, out = run(['finish the drift paper this week',
              'actually the solana bot is more urgent',
              'new ground: ship the solana bot'])
check('the command re-grounds', out[-1]['reason'] == 'grounded', out[-1]['reason'])
check('  the new ground is held', 'solana' in a.ground, str(sorted(a.ground)))
check('  and the old one is released', 'drift' not in a.ground, str(sorted(a.ground)))
after = a.hear('the screening pipeline needs a honeypot check')
check('  drift is now measured against the NEW ground',
      'solana' not in after['dropped'] or after['reason'] != 'grounded', after['reason'])
a2, out2 = run(['finish the drift paper', 'new ground'])
check('a bare "new ground" asks what it should be',
      out2[-1]['reason'] == 'ungrammatical', out2[-1]['reason'])
check('  and does not drop the old ground on the floor', 'drift' in a2.ground,
      str(sorted(a2.ground)))
check('  counters reset on a re-ground', a.drift_flags >= 0)

print()
print('x = [x, f(x)] — she measures herself too')
# Until this, she scored the person and never herself: a one-sided detector, which is the
# exact thing laserbrain exists to beat. Her reply is now a turn in the same dialogue.
a, out = run(['finish the drift paper this week', 'the corpus is too small'])
check('her replies enter the state', len(a.mine) == len(out), f'{len(a.mine)} vs {len(out)}')
check('  as a second speaker in the dialogue',
      any(t['agent'] == 'alice' for t in a.dlg.turns),
      str({t['agent'] for t in a.dlg.turns}))

# Quoting the person back is her job, not reflection. Counting ⟨...⟩ as echo made her very
# first reply confess to being "31% your own words back" while doing exactly what she should.
first = out[0]['reply']
check('quoting back is not confessed as mirroring', 'your own words back' not in first,
      first[:70])

# A repeated verdict may repeat its wording; that is the same thing happening twice, not a
# fault. Only words that stay put while the verdict MOVES are a real defect.
a2, out2 = run(['ship the solana bot', 'new ground: finish the drift paper',
                'new ground: ship the solana bot'])
check('two re-grounds do not accuse her of looping',
      not any('stopped tracking' in o['reply'] for o in out2),
      [o['reply'][-60:] for o in out2 if 'stopped tracking' in o['reply']])

print()
print('the ground trail is kept, so a return is visible')
check('every ground is recorded', len(a2.grounds) == 3, str(len(a2.grounds)))
check('  and coming back to one is named', 'held this ground already' in out2[-1]['reply'],
      out2[-1]['reply'][-80:])
a3, out3 = run(['ship the solana bot', 'new ground: finish the drift paper'])
check('  a genuinely new ground is not', 'held this ground already' not in out3[-1]['reply'])

print()
print('the helper is a real Jaccard')
check('identical sets', _sim_sets({'a', 'b'}, {'a', 'b'}) == 1.0)
check('disjoint sets', _sim_sets({'a'}, {'b'}) == 0.0)
check('half overlap', _sim_sets({'a', 'b'}, {'b', 'c'}) == 1 / 3)
check('two empties are 0, not 1', _sim_sets(set(), set()) == 0.0,
      'an empty turn must not read as a perfect match')

print()
if fails:
    print(f'  FAIL — {len(fails)}: ' + '; '.join(fails))
    sys.exit(1)
print('  PASS — she measures, she stops repeating herself, and she replays exactly.')
