# Engine

The platform-neutral core: the adaptation-run scheduler and its SQLite event
store, operator discovery, the failure-recovery loop, and the decider
interface. Verified to carry no vendor, runtime, or cluster references — the
tests guard that.

Reach it through its Python API (`engine.scheduler`, `engine.discovery`,
`engine.recovery`, `engine.brain`); it does not execute platform actions
itself.
