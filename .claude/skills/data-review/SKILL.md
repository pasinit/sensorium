---
name: data-review
description: Review generated sensorium episodes by eye and by numbers before calling a data step done or showing a sample to the supervisor - frames, behaviour, touch and contact stats, sound, speed and size. Use after generating or regenerating any sample, after changing a body, scene or policy, and before every report that includes data.
---

# Data review

Tests check that data is well-formed, not that it is worth learning from. The
first H1 sample passed every gate with 64 px frames in which fingers were 1-2
px, and a `hold` policy that touched nothing in 100% of steps. Look first.

## 1. Look at the frames

- Save a strip of frames (start, middle, end) from every variant and look at
  it (Read the PNG). Use `Episode.load`, which decodes the stored video.
- Can you see what matters: hands and fingers, the object being touched,
  contact, the room? Is anything clipped, black, flickering or repeated?
- Do episodes differ (appearance, layout) where they should?

## 2. Check the behaviour

- Does the policy do what its name says? Babbling should engage objects, not
  wave in the air; an expert should succeed.
- Per episode, from `meta` and `privileged`: `touch_rate`,
  `object_contact_rate`, `table_contact_rate`, falls, end reason. Flag any
  episode with no object contact, or any variant whose rates are far from the
  others.
- Physical sanity: peak contact forces (single-substep spikes of hundreds of
  N mean slamming), objects leaving the table, joints at their limits.

## 3. Check the other senses

- Touch charts: are the right regions active when the video shows contact?
- Sound: listen to one episode per variant; onsets should match contacts.
- Proprio and actions: smooth where expected, inside limits.

## 4. Check cost

- Steps per second per process, and wall time for the sample.
- Bytes per step on disk; extrapolate to the planned volume (~20M steps) and
  compare with the storage budget.

## 5. Write it down

Put the numbers in the PR description and the relevant doc. In the report to
the supervisor, say plainly what is good and what is not, with a link they
can open on a phone: publish the sample's `index.html` (with its `pages/`
files) as a private artifact.
