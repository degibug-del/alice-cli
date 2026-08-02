#!/usr/bin/env python3
"""Alice — a chatbot with no model in it.

WHAT SHE IS, AND WHAT SHE REFUSES TO BE

Alice does not answer questions. She cannot: there is no language model here, and a
deterministic system that pretended to answer would be doing what ELIZA did in 1966 —
reflecting your own words back with enough grammar to feel understood, and understanding
nothing. That trick works, which is the problem with it. People confided in ELIZA.

So Alice does the opposite of faking comprehension. She MEASURES the conversation and
reports the measurement. Every reply is a true statement about the structure of what you
just said, computed from it:

  · which words are new since the shared goal was set
  · which words you have stopped saying
  · how much this turn echoes the last one
  · whether the distance you report is closing

None of that requires understanding, all of it is checkable, and a model could not do it
more honestly. What she gives back is a mirror with a grammar — useful in the way a mirror
is useful, and no more.

POWERED BY LASERBRAIN

The control layer is `laserbrain._Dialogue`, the same object that scores multi-agent teams.
The first thing you say becomes the frozen shared goal, and every turn afterwards is scored
against it. Alice does not get to revise that goal; neither do you, without saying so. The
verdicts — grounded, topic-drift, echo-spiral, deliberation-stall — choose which of her
replies is true right now, so what she says is downstream of a measurement rather than of a
template picked at random.

DETERMINISM IS THE POINT

The same conversation always produces the same replies. There is no sampling, no
temperature, no seed. `test_alice.py` asserts this, because it is the property that makes
her auditable: if she says something surprising, you can replay it exactly.
"""
import hashlib
import os
import pathlib
import re
import sys

SDK = pathlib.Path(os.environ.get('LASERBRAIN_SDK') or
                   pathlib.Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/'
                                         'phronesis/laserbrain-sdk')
sys.path.insert(0, str(SDK))
try:
    from laserbrain import norm, PUBLISHED
    from laserbrain import _Dialogue
except Exception as e:                                          # noqa: BLE001
    raise SystemExit(
        f'Alice needs laserbrain. Tried {SDK} and got: {e}\n'
        'Set LASERBRAIN_SDK to the checkout, or `pip install laserbrain`.')

# `norm` drops the usual 30 stopwords; these are the ones that SURVIVE it and still carry
# no subject. Every entry here was observed being reported as "new" in the first real
# conversation — "new: ⟨ok, back, if, first⟩" is technically true and worthless, which is
# the failure this file's own docstring warns about, committed on the first try.
_THIN = {
    'think', 'know', 'want', 'like', 'thing', 'realli', 'just', 'realiz', 'mayb', 'maybe',
    'ok', 'okay', 'back', 'first', 'if', 'what', 'about', 'instead', 'should', 'would',
    'could', 'not', 'say', 'anyth', 'anything', 'enough', 'big', 'small', 'still', 'yet',
    'honestly', 'ran', 'run', 'get', 'got', 'make', 'need', 'go', 'do', 'be', 'have',
    'more', 'less', 'much', 'very', 'some', 'any', 'now', 'then', 'here', 'there',
}


# The phrasings she offers, and the ones a person reaches for instead.
_REGROUND = re.compile(r'^\s*(new ground|reground|re-ground|new goal|change the goal)\b[:,]?',
                       re.I)


def _sim_sets(a, b):
    """Jaccard over two token sets — the same measure laserbrain scores goals with."""
    if not a and not b:
        return 0.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _pick(options, *salt, avoid=None):
    """Choose deterministically from `options`, keyed by the content.

    Not random. The same turn in the same state always selects the same phrasing, which is
    what makes a transcript replayable — and it still avoids the flatness of one fixed
    string per verdict, because the key changes as the conversation does.

    `avoid` steps to the next option when the choice would repeat the previous reply.
    Content-keying alone is a coin flip across two options, and a bot that says the same
    paragraph twice running is doing the thing it exists to name.
    """
    h = hashlib.sha256('␟'.join(str(s) for s in salt).encode()).digest()
    i = h[0] % len(options)
    if avoid is not None and len(options) > 1 and options[i] == avoid:
        i = (i + 1) % len(options)
    return options[i]


def _quote(tokens, limit=4):
    """A few tokens, in a stable order, for showing back to the person."""
    return ', '.join(sorted(tokens)[:limit])


# ── questions she can answer truthfully ───────────────────────────────────────
# A deterministic chatbot has exactly one honest source of answers: facts about the
# conversation it is in. She holds the ground, every turn verbatim, the echo trail and the
# distance trail — so "what is the ground", "what did I say about X" and "am I repeating
# myself" have real answers, computed, not generated.
#
# Everything else gets a refusal that names what she CAN do. The temptation is to produce
# something agreeable for an unanswerable question, which is precisely the ELIZA move this
# whole design exists to avoid: an answer nobody can check, that feels like understanding.
_Q = {
    'ground':  re.compile(r"\b(what|which)\b.*\b(ground|goal|errand|task)\b|^ground\b", re.I),
    'recall':  re.compile(r"\b(what did i|did i ever|have i)\b.*\b(say|said|mention|talk)\b", re.I),
    'repeat':  re.compile(r"\b(am i|have i been)\b.*\b(repeat|circl|loop|echo)", re.I),
    'dropped': re.compile(r"\b(what).*\b(drop\w*|stopp\w*|gone|left out|forgot\w*|missing)\b", re.I),
    'far':     re.compile(r"\b(how far|how much closer|any progress|am i getting)\b", re.I),
    'self':    re.compile(r"\b(what|who) are you\b|\bwhat do you do\b|\bcan you (help|answer)\b", re.I),
    'covered': re.compile(r"\bwhat have (we|i) (covered|said|been|discussed|talked)\b", re.I),
    'social':  re.compile(r"^\s*(hi|hello|hey|yo|thanks|thank you|ty|cheers)\b[.! ]*$", re.I),
}
_TOPIC = re.compile(r"\babout\s+([a-z0-9' -]{2,40})", re.I)
# A question aimed at her, rather than a thought said aloud. Both end in "?", so the tell is
# a second person pronoun or a bare factual opener with no first person in it.
_UNANSWERABLE = re.compile(r"\?\s*$", re.I)

# How alike two grounds must be to count as the same one returned to. Deliberately NOT
# borrowed from the calibration: goal_min (0.30) is the floor for "still the same errand",
# which is a different question from "this is the errand you already abandoned once", and
# unifying them because both are similarities is how two thresholds that share a number
# come to share a bug. grammar.json makes the same warning about settled_max and
# self_report_min both being 0.15 — coincidence only.
_SAME_GROUND = 0.75

# The ⟨...⟩ spans: tokens she is deliberately quoting back, and the parenthetical notes she
# appends about herself. Neither is her own voice, so neither counts when she measures
# whether she is reflecting or looping.
_QUOTED = re.compile(r'⟨[^⟩]*⟩|\([^)]*\)')


def _answerable():
    return ('the ground · what you said about a thing · whether you are repeating yourself · '
            'what you have dropped · how far you have come · what we have covered')


class Alice:
    """One conversation. Feed it turns; it returns what it measured and what it says."""

    def __init__(self, calibration=None):
        self.cal = calibration or PUBLISHED
        self.dlg = None
        self.ground = set()
        self.last_user = set()
        self.history = []          # every user turn, as token sets
        self.log = []              # ...and verbatim, so recall can quote rather than paraphrase
        self.said = []
        self.mine = []             # her own replies, as token sets — she is a speaker too
        self.grounds = []          # every ground she has held, in order
        self.dists = []            # the person's distances only, so her turns cannot skew them
        self.turns = 0
        self.drift_flags = 0       # how many times drift has been named without a re-ground
        self._last_reason = None   # so a repeated verdict is not mistaken for a loop

    # ── the measurement ──────────────────────────────────────────────────────
    def hear(self, text, distance=5):
        """Take one user turn. Returns a dict: what was measured, and Alice's reply.

        `distance` is the person's own estimate of how far they are from what they came
        for. It is unanchored — their word, not a measurement — and is labelled as such
        wherever it is reported, the same way laserbrain labels a self-report.
        """
        text = (text or '').strip()
        now = set(norm(text))
        self.turns += 1

        # SHE OFFERS THIS, SO IT HAS TO WORK. The re-ground reply tells the person to say
        # "new ground" — and until this existed, saying it did nothing at all. That is the
        # precise failure the drift corpus documents: an instrument that instructs you to
        # use an affordance, receives the attempt, and silently ignores it teaches you the
        # affordance is broken. Adoption of `parent_goal` sat at 0.2% for exactly this
        # reason. An offer with no handler behind it is worse than no offer.
        if self.dlg is not None and _REGROUND.search(text):
            rest = _REGROUND.sub(' ', text).strip(' :,—-')
            fresh = set(norm(rest))
            if not fresh:
                return self._out('ungrammatical', text, now, 0.0,
                                 'Say what the new ground is, in the same breath — '
                                 '"new ground: ..." — and I will hold that instead.')
            old = _quote(self.ground, 3)
            self.dlg = _Dialogue(rest, self.cal)
            self.ground = fresh
            self.grounds.append(fresh)
            self.history, self.drift_flags = [fresh], 0
            self.last_user = fresh
            seen = self._ground_cycle(fresh)
            back = (f' You held this ground already, at re-ground {seen} — you have come back '
                    f'to it.' if seen else '')
            return self._out('grounded', rest, fresh, 0.0,
                             f'Re-grounded. ⟨{old}⟩ is released; ⟨{_quote(fresh, 6)}⟩ is what '
                             f'I hold now, and I will measure against exactly that.{back}')

        if self.dlg is None and _Q['social'].search(text):
            return self._out('answered', text, now, 0.0,
                             'Hello. Say what you came for — the first thing you say becomes '
                             'the ground, and I measure everything after it against that.')

        if self.dlg is None:
            if not now:
                return self._out('ungrammatical', text, now, 0.0,
                                 "I could not find a subject in that. Say what you came for.")
            self.dlg = _Dialogue(text, self.cal)
            self.ground = set(norm(text))
            self.grounds.append(set(self.ground))
            self.last_user = now
            self.history.append(now)      # the ground is a turn; turn 2 must be able to echo it
            self.log.append(text)
            reply = _pick([
                f'Held. The ground is ⟨{_quote(self.ground, 6)}⟩ — that is what I heard, '
                f'and I will measure everything after it against exactly that.',
                f'Ground set: ⟨{_quote(self.ground, 6)}⟩. I will not revise it, and I will '
                f'tell you when you do.',
            ], text, avoid=self.said[-1] if self.said else None)
            return self._out('grounded', text, now, 0.0, reply)

        # _Dialogue.step computes echo across DIFFERENT agents — `t['agent'] != agent` — so
        # with a single speaker it is always 0.00 and echo-spiral can never fire. Verified on
        # the first transcript: two near-identical turns both scored 0. For a chatbot the
        # interesting echo is the person repeating THEMSELF, so it is computed here.
        #
        # `restated_goal` is likewise not passed every turn. It means "I am declaring a new
        # goal", and passing the raw turn made every low-overlap sentence a declaration —
        # which fired topic-drift on five turns running and produced the same sentence five
        # times. That is the corpus's own 75% goal-drift chain, reproduced in a chatbot.
        self.log.append(text)
        refusal = None
        if _UNANSWERABLE.search(text) and self._answer(text, distance) is None:
            refusal = ('I cannot answer that — there is no model in me to answer it with. '
                       'What I can tell you: ' + _answerable() + '. Measuring it anyway:')

        answered = self._answer(text, distance)
        if answered is not None:
            self.said.append(answered)
            return {'turn': self.turns, 'reason': 'answered', 'echo': 0.0,
                    'ground': sorted(self.ground), 'new': [], 'dropped': [],
                    'reply': answered}

        v = self.dlg.step('you', text, distance)
        echo = max((_sim_sets(now, prev) for prev in self.history[-3:]), default=0.0)
        self.history.append(now)
        reason = self._verdict(v, now, echo)
        reply = self._reply(reason, text, now, echo, distance)
        if refusal:
            reply = f'{refusal} {reply}'
        self.last_user = now
        return self._out(reason, text, now, echo, reply)


    def _answer(self, text, distance):
        """A true answer, or None. Never a plausible one.

        Returning None hands the turn back to the measurement path, so anything she cannot
        answer is still scored rather than silently absorbed.
        """
        if _Q['social'].search(text):
            if not self.dlg:
                return 'Hello. Say what you came for — the first thing you say becomes the ground.'
            return f'Still here, holding ⟨{_quote(self.ground, 4)}⟩. Go on.'

        if _Q['self'].search(text):
            return ('I measure this conversation, I do not answer questions about the world — '
                    'there is no model in me to do it with. I can tell you: '
                    + _answerable() + '.')

        if not self.dlg:
            return None

        if _Q['ground'].search(text):
            return (f'The ground is ⟨{_quote(self.ground, 8)}⟩, set on turn 1 and unchanged since. '
                    f'Say "new ground: ..." to move it.')

        if _Q['recall'].search(text):
            m = _TOPIC.search(text)
            if not m:
                return ('About what? Name the thing and I will quote you back — I keep every '
                        'turn verbatim.')
            want = set(norm(m.group(1)))
            hits = [line for line, toks in zip(self.log, self.history) if want & toks]
            if not hits:
                return (f'Nothing. ⟨{_quote(want, 3)}⟩ has not appeared in {len(self.log)} '
                        f'turn(s) — you have not raised it.')
            shown = hits[-2:]
            return ('You said: ' + ' … and … '.join(f'"{h}"' for h in shown)
                    + f' ({len(hits)} of {len(self.log)} turns mention it.)')

        if _Q['repeat'].search(text):
            if len(self.history) < 2:
                return 'Not yet — there is only one turn to compare against.'
                
            echoes = [_sim_sets(self.history[i], self.history[i - 1])
                      for i in range(1, len(self.history))]
            worst = max(echoes)
            return (f'Highest overlap between consecutive turns is {worst:.0%}'
                    + (f' — above the {self.cal.echo_min:.0%} floor, so yes.'
                       if worst >= self.cal.echo_min else
                       f', under the {self.cal.echo_min:.0%} floor. Not by my measure.'))

        if _Q['dropped'].search(text):
            latest = self.history[-1] if self.history else set()
            gone = {x for x in self.ground - latest if x not in _THIN}
            return (f'⟨{_quote(gone, 6)}⟩ — in the ground, absent from your last turn.'
                    if gone else 'Nothing. Your last turn still carries the whole ground.')

        if _Q['far'].search(text):
            dh = self.dists
            if len(dh) < 2:
                return f'You have said {distance} once. Nothing to compare it to yet.'
            return (f'Your distance went {dh[0]} → {dh[-1]} over {len(dh)} turns. '
                    f'{"Closing." if dh[-1] < dh[0] else "Not closing."} '
                    f'That is your own estimate, not a measurement.')

        if _Q['covered'].search(text):
            seen = set()
            for toks in self.history:
                seen |= toks
            seen = {x for x in seen if x not in _THIN}
            return (f'{len(self.log)} turns. Ground ⟨{_quote(self.ground, 4)}⟩; '
                    f'everything raised: ⟨{_quote(seen, 12)}⟩.')
        return None


    def _self_check(self, reply, user_now, reason=None):
        """x = [x, f(x)] — her own reply is measured and folded back into the state.

        Until this existed she scored the person and never herself, which is the one-sided
        detector laserbrain was built to beat: a monitor watching only the ground can never
        name the case where the READING is what is cycling. Two things become visible only
        once her turn is a turn:

          MIRROR  her reply overlapping the sentence that prompted it. That is the ELIZA
                  move — reflecting your words back with enough grammar to feel understood
                  — and it is the specific failure this whole design claims not to make.
                  She cannot claim that and also not check it.
          LOOP    her reply overlapping her own previous replies. She tells people that
                  repetition is not progress; the same standard has to apply to her.

        Her turn also enters the shared _Dialogue as a second speaker, which is what that
        object was written for — its echo term compares ACROSS agents, and with one speaker
        it was dead code.
        """
        # Her ⟨quoted⟩ spans are the person's words ON PURPOSE — showing back exactly what
        # was heard is the honesty the whole design rests on. Counting them as echo made the
        # very first reply confess to being "31% your own words back", which is true, and is
        # her doing her job. Only what she says in her OWN voice can be reflection.
        mine = set(norm(_QUOTED.sub(' ', reply)))
        mirror = _sim_sets(mine, user_now)
        loop = max((_sim_sets(mine, prev) for prev in self.mine[-3:]), default=0.0)
        self.mine.append(mine)
        if self.dlg is not None:
            self.dlg.step('alice', reply, self.dists[-1] if self.dists else 5)

        # A REPEATED VERDICT IS ALLOWED TO REPEAT ITS WORDS. Two re-grounds in a row produce
        # the same frame because the same thing happened twice, and confessing to "looping"
        # there would be a false alarm dressed as rigour — the note would fire on her working
        # correctly. What is NOT allowed is her words staying put while the verdict moves:
        # that means her vocabulary has stopped tracking her own state, and it is the only
        # version of this she can honestly call a fault.
        same_verdict = reason is not None and reason == self._last_reason
        self._last_reason = reason

        notes = []
        if mirror >= self.cal.echo_min:
            notes.append(f'(I am {mirror:.0%} your own words back — that is reflection, '
                         f'not reading. Discount it.)')
        if loop >= self.cal.echo_min and not same_verdict:
            notes.append(f'(That is {loop:.0%} what I already said, on a different verdict. '
                         f'My words have stopped tracking my own state.)')
        return ' '.join(notes)

    def _ground_cycle(self, fresh):
        """Has this ground been held before? The ground trail, checked before the readings.

        laserbrain checks the GROUND trail first and falls back to the reading trail,
        because a person returning to a goal they abandoned and a bot repeating a verdict
        over a goal that moved are different findings. With one overwritten ground the first
        was undetectable — a re-ground back to where you started looked like a fresh start.
        """
        for i, old in enumerate(self.grounds[:-1]):
            if _sim_sets(fresh, old) >= _SAME_GROUND:
                return i + 1
        return None

    def _verdict(self, v, now, echo):
        """Alice's own reading, layered over the dialogue's.

        Two things the raw verdict cannot do for a two-party chat: notice a person echoing
        themself, and stop saying "you have drifted" to someone who has heard it. After the
        ground has been named as lost twice without a re-ground, the honest move is to stop
        accusing and offer to move the ground — repeating an unheeded verdict is what the
        corpus calls a goal-drift chain, and it is useless to the person in it.
        """
        if v['reason'] == 'ungrammatical':
            return 'ungrammatical'
        overlap = _sim_sets(now, self.ground)
        if echo >= self.cal.echo_min and len(self.history) > 1:
            return 'echo-spiral'
        if overlap < self.cal.goal_min:
            self.drift_flags += 1
            return 'topic-drift' if self.drift_flags <= 2 else 'reground-offer'
        self.drift_flags = 0
        return v['reason'] if v['reason'] in ('deliberation-stall',) else 'advancing'

    def _out(self, reason, text, now, echo, reply):
        note = self._self_check(reply, now, reason)
        if note:
            reply = f'{reply} {note}'
        self.said.append(reply)
        new = {t for t in now - self.ground if t not in _THIN}
        gone = {t for t in self.ground - now if t not in _THIN}
        return {'turn': self.turns, 'reason': reason, 'echo': round(echo, 2),
                'ground': sorted(self.ground), 'new': sorted(new), 'dropped': sorted(gone),
                'reply': reply}

    # ── the reply, chosen by the verdict ─────────────────────────────────────
    def _reply(self, reason, text, now, echo, distance):
        new = {t for t in now - self.ground if t not in _THIN}
        gone = {t for t in self.ground - now if t not in _THIN}
        repeat = now & self.last_user

        if reason == 'ungrammatical':
            return _pick([
                'There is no position in that I can measure. Say one thing you want.',
                'I cannot score that — it has no subject. What are you actually after?',
            ], text, avoid=self.said[-1] if self.said else None)

        if reason == 'topic-drift':
            left = _quote(gone) or 'the ground'
            came = _quote(new) or 'something else'
            return _pick([
                f'You have left ⟨{left}⟩ and arrived at ⟨{came}⟩. Is this the same errand, '
                f'or a new one? If it is new, say so and I will re-ground.',
                f'That is a different subject. ⟨{left}⟩ has dropped out; ⟨{came}⟩ is what you '
                f'are on now. Do you want to keep the old ground or replace it?',
            ], text, avoid=self.said[-1] if self.said else None)

        if reason == 'reground-offer':
            return _pick([
                f'You have been off ⟨{_quote(gone, 3) or "the ground"}⟩ for three turns now. I am '
                f'going to stop calling that drift — say "new ground" and I will hold '
                f'⟨{_quote(new, 3) or "what you just said"}⟩ instead.',
                f'Twice now I have said the ground is ⟨{_quote(gone, 3) or "where you started"}⟩ and twice '
                f'you have gone elsewhere. The ground is probably wrong. What should it be?',
            ], text, avoid=self.said[-1] if self.said else None)

        if reason == 'echo-spiral':
            return _pick([
                f'We are circling ⟨{_quote(repeat)}⟩ — the last few turns are mostly the '
                f'same words. What have you not said yet?',
                f'That echoes your previous turn at {echo:.0%}. Repetition is not progress; '
                f'name one thing that is different now.',
            ], text, avoid=self.said[-1] if self.said else None)

        if reason == 'deliberation-stall':
            return _pick([
                f'You have said {distance} for a while now. What would have to be true for '
                f'that number to be lower?',
                'The distance is not moving. Either the goal is wrong or the approach is — '
                'which one do you want to look at?',
            ], text, avoid=self.said[-1] if self.said else None)

        # grounded / advancing
        if new:
            return _pick([
                f'New since the ground: ⟨{_quote(new)}⟩. Say more about that — it is the '
                f'part I have not measured before.',
                f'⟨{_quote(new)}⟩ is new. Does it serve ⟨{_quote(self.ground, 3)}⟩, or '
                f'replace it?',
            ], text, avoid=self.said[-1] if self.said else None)
        if gone:
            return _pick([
                f'You have stopped saying ⟨{_quote(gone)}⟩. Dropped, or just unsaid?',
                f'⟨{_quote(gone)}⟩ has gone quiet since the ground. Is it handled?',
            ], text, avoid=self.said[-1] if self.said else None)
        return _pick([
            'That is the same ground, said again. What is in the way?',
            'Still on the ground and nothing new in it. What would move this?',
        ], text, avoid=self.said[-1] if self.said else None)
