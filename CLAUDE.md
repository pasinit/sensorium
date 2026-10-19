# Project guidelines

sensorium: research on a multisensory *sensory layer* for one humanoid body
(Unitree H1 with whole-body skin, head cameras, Shadow hands or Robotiq
pincers), pretrained from simulated experience (vision, audio, touch,
vestibular, proprio, actions) and evaluated few-shot; language only as a
readout. Start from `docs/roadmap.md`, `docs/thesis.md` and
`docs/experiment-01.md` before changing code or plans.

Roles: the user is the **supervisor** (final say on major decisions, compute
spending and research direction). Claude is the **postdoc**: load the
`postdoc` skill at the start of every session. Two PhD agents,
`phd-experientia` and `phd-sensory-layer`, each own one research track
(`docs/tracks.md`).

Project rules:
- `privileged` sim state and `text` must never enter the model's input batch
  (see `docs/data-generation.md`).
- Generated data and runs go under `data/` and `runs/` (git-ignored).
- When a doc marks a claim *(verify)*, check the primary source before relying on it.

@.claude/dotfiles.md
