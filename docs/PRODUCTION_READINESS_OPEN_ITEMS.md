# Production-readiness open items — NOT closeable by engineering

**Status: all four OPEN. None has a named owner.**

These are the blockers that no implementation task closes. They are recorded
here, with explicit owner fields, so that "nobody was assigned" is a visible
fact in the repository rather than something that quietly evaporates between
one work item and the next. Every previous pass deferred these to "whoever
owns compliance"; nobody owns compliance yet.

The system currently auto-fires a CRITICAL alert naming a specific person, to
an officer, on the basis of a face match, at a threshold that has never been
validated against real data. That is a materially different risk profile from
a hackathon demo, and these four items are what separates the two.

---

## 1. Data retention for biometric and journey data

| Field | Value |
|---|---|
| **Owner** | ⬜ UNASSIGNED |
| **Status** | OPEN |
| **Blocks** | Go-live |

Face embeddings, body (ReID) embeddings, and journey/movement histories are
biometric and location data about identifiable people.

What exists in code:
- `REID_RETENTION_DAYS=90`, `JOURNEY_RETENTION_DAYS=30`, `AUDIT_RETENTION_DAYS=365`
  in `.env`. Every one is commented as a placeholder.
- Working deletion jobs: `backend/scripts/reid_retention_job.py` (daily 02:00)
  and `backend/scripts/retention_job.py` (daily 03:00).

What does not exist:
- A legally-determined retention period for each data class, under the DPDP
  Act, applicable MHA guidelines, and State police IT policy.
- A named person accountable for those numbers being right.
- A documented, exercisable subject-deletion path (the jobs delete on age;
  there is no "delete this person's data on request" flow).

**The mechanism is built. The numbers in it are engineering guesses, and
changing them is a one-line `.env` edit — which is exactly why nobody has
had to decide what they should be.**

---

## 2. Formal approval to run face recognition against a police watchlist

| Field | Value |
|---|---|
| **Owner** | ⬜ UNASSIGNED |
| **Status** | OPEN |
| **Blocks** | Any deployment against a real watchlist |

Jurisdiction-specific and not an engineering call. Required before any
`watchlist_persons` row contains a real person.

Until approval exists, the watchlist should hold only synthetic test entries.
There is currently no technical control enforcing that.

---

## 3. Bias / disparate-impact testing

| Field | Value |
|---|---|
| **Owner** | ⬜ UNASSIGNED |
| **Status** | OPEN |
| **Blocks** | Go-live |

`backend/scripts/reid_validate.py --mode face` produces an **overall** false
positive rate. It does not, and cannot, tell you whether that rate differs
across demographic groups — which is the question that determines whether the
system disproportionately subjects particular communities to being stopped.

This requires:
1. A dataset design decision specifying the demographic groups relevant to
   Gujarat deployment contexts.
2. Labeled data carrying those attributes.
3. Analysis by someone qualified to run and interpret it.

**A single ROC curve does not answer this.** Every generated validation report
carries this disclaimer so it cannot be overlooked, but a disclaimer in a
report is not a completed analysis.

---

## 4. Written human-in-the-loop operating procedure

| Field | Value |
|---|---|
| **Owner** | ⬜ UNASSIGNED |
| **Status** | OPEN — draft below, unapproved |
| **Blocks** | Go-live |

Right now, nothing stops a `WATCHLIST_FACE_MATCH` alert from being acted on
before an officer has visually confirmed it. The side-by-side reference/live
photo compare exists in `AlertFeed.jsx`, and the "mark false positive" control
exists — but both are *after-the-fact* affordances. Using them is not
required, and no record is kept of whether anyone looked.

### Draft procedure — NOT APPROVED, for the owner to revise

> **On a WATCHLIST_FACE_MATCH alert, the officer MUST:**
>
> 1. Open the side-by-side comparison and visually compare the live capture
>    against the watchlist reference photo before taking any action.
> 2. Treat the match as **unconfirmed** until that comparison is done. The
>    alert is a prompt to look, not a determination of identity.
> 3. Where the comparison is inconclusive — poor angle, low resolution,
>    partial occlusion, motion blur — treat it as **not a match** and record
>    it as a false positive. Inconclusive is not weak evidence for; it is
>    absence of evidence.
> 4. Record the outcome (confirmed / false positive) before the alert is
>    closed.
>
> **The officer MUST NOT:**
>
> - Detain, stop, search, or approach a person solely on the basis of this
>   alert without the visual confirmation in step 1.
> - Treat the confidence percentage as a probability that the person is who
>   the system says they are. It is a cosine similarity between two
>   embeddings, at a threshold that has not been validated against real
>   footage (see item 3 and `reports/face_watchlist_validation_*.md`).
> - Rely on the alert where the system reports the embedder is in stub mode
>   (`is_stub: true` in the alert metadata). Those matches are meaningless.

### What engineering can contribute, and what it cannot

Contributed (see `AlertFeed.jsx`): the draft procedure is surfaced on the
alert itself, so it is in front of the officer at the moment of decision
rather than in a document they have never read.

**Not contributed, and not closeable here:** whether the procedure is correct,
whether it is mandated, whether compliance is enforced or recorded, and what
happens when it is not followed. Those are decisions for whoever owns this
deployment operationally. An affordance in a UI is not a procedure, and this
item stays OPEN until someone with the authority to mandate it signs it off.

---

## Review

Re-read this file at every go/no-go checkpoint. An item is closed only when
its Owner field names a real person and its Status records their decision —
not when the surrounding code is finished.
