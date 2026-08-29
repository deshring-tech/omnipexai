# Recruiting & Facilitation Kit

Everything needed to run participant sessions consistently and without bias.
Read PROTOCOL.md first; this file is the operational half.

**The cardinal rule:** the participant must not know which workflow is "the
product," must not get correctness feedback mid-session, and must not be told the
product name until the debrief. The harness is already neutral (Workflow A/B, no
verdicts printed); everything you *say* must match it.

---

## 1. Recruiting post (Reddit r/RAG, r/LangChain, Discords, MLOps Slack)

> **[Study] How do you debug eval regressions? 30-min session, $25 gift card**
>
> I'm running a small study on how engineers debug quality regressions in
> LLM/RAG pipelines — specifically what happens after a commit changes several
> things and an eval metric drops.
>
> Format: 30-minute video call, screen-share, you work through a few short
> debugging scenarios in a terminal (everything provided, nothing to install if
> you prefer I drive). No preparation needed.
>
> Looking for people who have shipped a RAG or LLM pipeline and run evals on it
> (DeepEval, RAGAS, LangSmith, promptfoo, homegrown — anything).
>
> $25 gift card as a thank-you, and I'll share the aggregated findings with
> everyone who participates. DM me with a sentence about the last eval regression
> you had to chase down.

Notes:
- The "DM me with a sentence about your last regression" line **is** the first
  screener — people with a real story qualify and are pre-engaged.
- Do not mention Omnipex, "attribution," "a tool I built," or anything that
  signals which arm is yours.

## 2. Screener (qualify before scheduling)

Ask in DM; qualify if YES to 1–2 and ideally 3:

1. Have you built or maintained a RAG / LLM pipeline that reached users (or a
   serious internal deployment)?
2. Do you run evals on it (any framework, or homegrown scripts)?
3. Have you ever had a metric drop after a commit that changed **more than one
   thing** (model, prompt, chunking, retrieval params)? What did you do?
4. (Logistics) 30 minutes on a video call with screen-share this week or next?

Disqualify politely: students with no shipped pipeline, people who only use
ChatGPT/completion APIs without any retrieval/eval loop. Record the answer to #3
verbatim — those stories are market-research gold regardless of the session.

## 3. Session script (read the framing verbatim; improvising is how bias leaks in)

### Setup (before the call)
- [ ] `python -m valuetest.cli check` passes.
- [ ] Fresh terminal, font large enough for screen-share.
- [ ] `run --participant <initials> --penalty <current default>` ready to paste.
- [ ] Notes template (section 5) open on a second screen, not shared.

### Opening (verbatim)

> "Thanks for doing this. I'm comparing **two ways of debugging the same kind of
> problem**: a commit changed a few pipeline parameters, an eval metric moved,
> and the job is to figure out which change — if any — caused it.
>
> You'll see a handful of short scenarios. For each one the terminal assigns one
> of two workflows — A or B. **I'm not telling you yet which one I'm studying,
> and I won't react to anything you do.** There's no trick and no grade; work
> exactly as you would at your desk, and please think aloud as you go.
>
> One mechanical note: in Workflow A you can isolate a single change and re-run
> the evals. In real life that's a CI cycle, so the timer charges a realistic
> cost for each re-run. Spend re-runs the way you would spend real CI time.
>
> Any questions before we start?"

### During
- Silence except mechanics. Allowed: "you can enter a number or 's'." Not
  allowed: "are you sure?", "interesting choice", any reaction to answers.
- Note timestamps of hesitation, backtracking, and every think-aloud quote about
  trust or doubt. The quotes are the qualitative payload.
- If they ask "is this right?": "I can't say during the session — we'll go
  through everything at the end."

### Debrief (after the last scenario — now you may reveal)
Ask in this order, and get answers to Q1–Q3 **before** revealing which arm was
the product:

1. "Which workflow did you prefer, and why?" (their words, before your reveal)
2. "In Workflow B, did you trust the report? What would make you trust it more
   or less?"
3. "Walk me through the scenario you found hardest."
4. Reveal: which arm was the tool, what it does, the ground truth per scenario.
5. "The last time this happened to you for real — would this have changed what
   you did?" (listen for a specific story, not politeness)
6. The anger test: "If you had this in CI and it was taken away next month,
   would you care? How much?"
7. "Can I follow up when there's a version that runs on your pipeline?" — a YES
   here is a design-partner lead; log it.

## 4. Facilitator checklist

- [ ] Never run `analyze` on a shared screen.
- [ ] One participant = one `--participant` id; never reuse.
- [ ] Back up `valuetest/results/sessions.jsonl` after every session.
- [ ] Log qualitative notes within an hour, while memory is fresh.
- [ ] **Self-run kits (later): blind the build first** — `scenarios.py` currently
      contains `expected` and facilitator `note` fields in plaintext. Strip or
      encode them before the repo leaves your machine. Facilitated sessions are
      fine (participant never opens the source).
- [ ] After ~4 sessions, sanity-check arm balance per scenario in the results.

## 5. Per-session notes template

```
participant: ____   date: ____   source (subreddit/discord/network): ____
screener #3 story (verbatim):
scenario-by-scenario hesitations / quotes:
  s1: 
  s2: 
  s3: 
  s4: 
  s5: 
  s6: 
debrief:
  preferred workflow + why:
  trust in the report (what would raise/lower it):
  hardest scenario, their account:
  "would it have changed your real incident?":
  anger test (1-5 + quote):
  design-partner follow-up OK? (y/n, contact):
facilitator impressions (2 sentences max):
```
