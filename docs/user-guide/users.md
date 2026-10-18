---
description: "Signing in, the admin and viewer roles, managing users and the audit log."
---

# Users and roles

## Sign in and roles

Open `http://<host>:8090/dgxmm/` (or your proxy URL) and sign in. On first install the user is
`admin`, and the installer printed its password. It is also saved in `secrets/admin_password`. You must
pick a new password (at least 12 characters) before you can do anything else. The same applies to users
an admin creates or resets.

| Role | Can |
|---|---|
| **viewer** | See the dashboard, deployments (including each one's LiteLLM endpoint and model name), models, catalog, downloads, planner and processes, and run memory checks. Cannot see users, audit log, container logs, the LiteLLM page or virtual keys. |
| **admin** | Everything: deploy, start, stop, sleep, switch, reconfigure, remove, download, delete weights, benchmark, view container logs, configure LiteLLM and manage virtual keys, configure the decision model, manage users, read the audit log. |

After 5 failed sign-ins within a minute (per IP or per username), further attempts are refused for a
minute. Sessions last 12 hours from the last activity. Forgot the admin password? See
[operations](../operations/troubleshooting.md#forgot-the-admin-password).

## Users and audit log

**Users** (admins): add users with a role and a temporary password, change roles, reset passwords
(users must change the temporary password at next sign-in, and their sessions end), and delete users.
You cannot delete yourself or remove the last admin.

**Audit log** (admins): who did what and when. It covers sign-ins (including failed ones), deploys,
stops, switches, reconfigures, model name changes, downloads, weight deletions and clean-ups (also
refused and failed ones), benchmarks, LiteLLM settings, syncs and key changes, decision-model changes
(enable, disable, settings, image builds, model downloads, key rotations, automatic restarts and
feedback), user changes and watchdog actions. Actions run with the admin CLI are marked `"via": "cli"`.
Filter by action.

## Next steps

- [Settings](settings.md)
- [Application security model](../security/application.md)
