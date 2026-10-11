# API maintenance

- GitHub source: https://github.com/aguilardmarkanthony/nba-injuries-api
- Update code, tests, README, changelog, and RapidAPI docs/specifications together.
- Review changes and run pytest, Ruff, pip check, and dependency audit before release. Regenerate specifications with `python scripts/export_openapi.py` and verify the generated files are committed.
- Commit and push reviewed updates to GitHub for every API task, as requested by the user. Preserve v1 compatibility and do not commit credentials or customer/billing payloads.
- Render deploys `main`. Verify the deployed version and authenticated gateway behavior before publishing matching RapidAPI documentation. Keep origin URLs in the provider import specification and the gateway URL in the consumer specification.
- Propose pricing changes and paid hosting separately; do not change subscriber billing as a side effect of code maintenance.
