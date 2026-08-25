"""Interactive TUI for manual annotation review after auto-cleaning.

Module layout:
    models.py       -- pure dataclasses (EditRecord, ReviewedFile)
    journal.py      -- per-subject on-disk session log + reviewed-files
                       tracker; JSONL append-only for crash-recovery
    (upcoming)
    controller.py   -- state machine, unit-testable without a TTY
    apply_edits.py  -- corruption-safe batch application of pending
                       edits back to the EDF headers on disk
    tui.py          -- prompt_toolkit wiring
"""
