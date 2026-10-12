# RapidAPI plan settings

Updated October 12, 2026 (Asia/Singapore). The owner explicitly requested reducing the NBA API's free allowance from 30 to 5 requests.

| Public plan | Monthly subscription | Monthly request allowance |
| --- | ---: | ---: |
| BASIC | $0 | 5, hard limit |
| PRO | $9.99 | 500,000, hard limit |

BASIC is a recurring monthly allowance, not a one-time five-request trial. After the request quota is exhausted, the hard limit prevents further calls for that quota period. RapidAPI manages quota enforcement; no backend change is required.

RapidAPI preserves the previous plan for existing subscribers when a provider edits a plan. The new allowance applies to subscriptions to the updated BASIC plan. No existing subscriber was canceled, blocked or migrated. [RapidAPI plan documentation](https://docs.rapidapi.com/docs/hub-listing-monetize-tab)

The platform's separate `Rapid-free-plans-hard-limit` object remains 500,000 and does not replace the smaller BASIC Requests quota. Bandwidth settings remain separate: 10,240 MB included, then $0.001/MB. PRO pricing/quota and disabled ULTRA/MEGA settings were preserved.

## Verification

- In the NBA project's Public Plans editor, confirmed the prior Requests quota was 30, Monthly was selected and Hard Limit was selected.
- Changed only the BASIC Requests quantity to 5 and saved.
- Reopened the saved `5 / month` setting and verified BASIC, quantity 5, Monthly and Hard Limit before canceling the verification dialog.
- Verified the public pricing page shows BASIC at $0 with `5 / Month` and `Hard Limit`; PRO still shows $9.99 and 500,000/month. The owner's separate deprecated BASIC subscription remains visible with its prior 500,000/month allowance.
- Saved and reopened the consumer documentation; all 9,772 characters matched `docs/rapidapi.md` exactly, including the new allowance. Updated GitHub records. Backend version remains 2.4.0, with no endpoint/schema change.

This verifies the saved marketplace configuration. It does not claim a new-account exhaustion test: the owner's existing subscription retains its earlier plan.

[Public pricing](https://rapidapi.com/aguilardmarkanthony/api/nba-injuries-api/pricing) · [Provider settings](https://rapidapi.com/studio/api_d83a29e5-19c1-4794-b7e5-994f33225392/publish/monetize/public_plans)
