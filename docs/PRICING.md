# RapidAPI plan settings

## Current: portfolio review, October 12, 2026

The owner clarified that free access should let customers sample the API rather than support ongoing daily use. The temporary 100-call evaluation is reduced to five monthly requests. Paid pricing from the competitive review is retained.

| Public plan | USD/month | Requests/month | Rate limit |
| --- | ---: | ---: | --- |
| BASIC | $0 | 5, hard limit | 1/second |
| PRO | $5 | 500,000, hard limit | 3,600/hour |

Five calls support a brief sample of responses and filters, not a full test of every endpoint or daily monitoring. They renew monthly without automatic paid conversion. PRO remains $5 after the earlier reduction from $9.99, with its paid quota unchanged. The directly comparable [NBA Injuries Reports](https://rapidapi.com/nichustm/api/nba-injuries-reports/pricing) offers $5/500,000 and one free request. [BALLDONTLIE](https://docs.balldontlie.io/#account-tiers) includes injuries in its broader $9.99 All-Star package, not its free tier. These observed prices do not establish data or reliability parity.

Existing subscribers keep their previous plans until they switch; no account is forcibly migrated. All gateway calls can consume quota. Bandwidth remains 10,240 MB/month included, then $0.001/MB. A $5 subscription leaves $3.75 after RapidAPI's 25% marketplace fee, before hosting, support, payout costs, refunds and tax. Profitability remains unproven. [Marketplace fees](https://docs.rapidapi.com/docs/payouts-and-finance)

See the portfolio's [pricing review and verification](https://github.com/aguilardmarkanthony/api-catalog/blob/main/docs/PORTFOLIO-PRICING-REVIEW-2026-10-12.md). Backend version remains 2.4.0; this update changes pricing and documentation. ULTRA and MEGA remain disabled. The owner's grandfathered test subscription does not verify new-account quota exhaustion.

## Earlier same-day change (superseded)

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
