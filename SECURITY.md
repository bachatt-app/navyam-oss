# Security notes & hardening backlog

Living doc for security posture and follow-ups. Add items as they come up; check
them off when done.

## Postgres (`bachatt-postgres-prod`, DB `navyam_gpt`) — network access

**Current state (2026-09-06):** public access enabled with an IP allowlist, plus
an `AllowAllAzureServices` firewall rule (start/end `0.0.0.0`) so the App Services
`navyam-gpt` (consumer) and `your-admin-webapp` can reach it.

**Why the broad rule exists:** Azure App Service uses a *pool* of ~32 default
outbound IPs (not a static IP), and that pool rotates on plan scale/move/reconfig.
The firewall previously allowlisted specific App Service IPs; when the pool rotated,
every DB call began timing out and admin + consumer login broke (incident
2026-09-06). The `0.0.0.0` rule allows Azure-*internal* services only (not the
public internet) and still requires DB credentials, so it's a safe stopgap — but
it's broader than the curated allowlist we'd prefer.

### TODO — tighten to a static, narrow path (then drop the broad rule)

- [ ] **Static egress IP for the App Services.** VNet-integrate `navyam-gpt` and
  `your-admin-webapp` (regional VNet integration) and route outbound through a **NAT
  Gateway with a static public IP**. Then allowlist that single IP on the Postgres
  firewall and **delete the `AllowAllAzureServices` (0.0.0.0) rule**. Result: one
  stable IP instead of a rotating ~32-IP pool.
- [ ] **Or, Private Endpoint to Postgres.** Put Postgres behind a Private Endpoint
  in the same VNet the App Services integrate with; disable public network access
  entirely. Most secure (no public exposure at all), but requires VNet plumbing and
  private DNS. Prefer this if we're going to run a VNet anyway.
- [ ] **Audit the existing allowlist** (`az postgres flexible-server firewall-rule
  list -g your-aks-rg -s bachatt-postgres-prod`) and prune stale personal/temp IPs
  (e.g. `ankur-local`, `bq-sync-prep-temp`) once the app path is static.
- [ ] **Least-privilege DB roles.** Confirm the app connects as a scoped role
  (`navyam_gpt_app`) with only the grants it needs (no superuser); the consumer
  `consumer_chats` table should be writable only by that role.

## Other

- [ ] navyam.ai serving: serve.py has no auth beyond the Caddy `@trusted` App
  Service IPs + bearer key; revisit if the endpoint is ever exposed more broadly.
- [ ] Rotate the Postgres app password (`~/Bachatt/devops/azure/navyam-gpt-secrets/db.txt`)
  periodically; it's currently long-lived.
