# Restoring this session after a pod rebuild

`MEMORY.md` and `stick-pivot-epic-5238.md` here are Claude's memory files.
The conversation transcript itself is **not** in this repo — see below.

## Memory files

    mkdir -p ~/.claude/projects/-home-senuser/memory
    cp MEMORY.md stick-pivot-epic-5238.md ~/.claude/projects/-home-senuser/memory/

## The transcript (run this yourself BEFORE deleting the pod)

Claude could not copy it into the repo. From the live pod:

    cd /home/senuser/.claude/projects/-home-senuser
    gzip -9 -c 0eea9bcf-770d-4b0b-920d-af1245b6c115.jsonl > /tmp/session.jsonl.gz

Then either keep `/tmp/session.jsonl.gz` somewhere outside the pod, or add it
to this branch and push:

    cp /tmp/session.jsonl.gz <repo>/analysis/epic5238/session/
    git add analysis/epic5238/session/session.jsonl.gz
    git commit -m "analysis: save the epic #5238 session transcript"
    git push origin epic5238-view-testing

13 MB raw, about 2 MB gzipped.

## To let Claude resume from it later

    gunzip -c session.jsonl.gz \
      > ~/.claude/projects/-home-senuser/0eea9bcf-770d-4b0b-920d-af1245b6c115.jsonl

Start Claude Code from the same working directory the session used
(`/home/senuser`) and resume. The project directory name is derived from the
working directory, so it must match: `-home-senuser`. If the new pod has a
different home, rename accordingly (`/home/foo` -> `-home-foo`).

## Faster than reading the transcript

`../HANDOFF.md` has the state, the measured numbers, the issue verdicts and
what is left. `.claude/skills/epic5238-view-testing/SKILL.md` has how to
re-run everything and how to bisect a stick-layout bug. Start there; the
transcript is only for the detail behind a specific decision.

Keep the transcript off any branch that goes upstream — it contains the whole
working session, including file contents and tool output.
