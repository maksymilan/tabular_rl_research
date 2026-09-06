Root one-off scripts were assigned to an explicit owner:

- current Spider evaluation adapters: `src/rl/scenarios/evaluation/spider_variant/`;
- pass@k resume entrypoint: `src/rl/scenarios/evaluation/resume_passk_shard.py`;
- historical rollout supervisors: `archive/experiments/legacy_202608/`.

New scripts must be placed beside the scenario or fixed module that owns them.
