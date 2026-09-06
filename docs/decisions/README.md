# Decision records

Decision records explain why the current system has its shape. They are not executable contracts;
use `docs/current/` for active behavior.

- `provenance_redesign.md`: provenance trust boundary and migration history. Compiler/enrichment
  implementation references inside it are historical.
- `process_reward_density.md`: formal motivation for dense, grounded process credit.
- `process_reward_design.md`: reward-design exploration; active implementation is
  `src/rl/objectives/process_credit.py`.
- `context_management_design.md`: origin of bounded context; SQL-compiled perception injection is
  superseded by causal online tool choice.
- `subtable_vs_memory.md`: historical state-ownership analysis.
- `table_text_handling.md`: future text/table integration considerations.
