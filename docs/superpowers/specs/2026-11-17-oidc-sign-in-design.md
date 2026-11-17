# Single sign-on with OpenID Connect — design

Date: 2026-11-17 · Status: implemented (`dgxmm/auth/oidc.py`, `dgxmm/api/sso.py`).

## Intent

DGXMM often runs behind a reverse proxy that already signs people in with an identity provider (IdP). DGXMM's own
password login then asks the same person to sign in a second time, with an account the IdP knows nothing about.
DGXMM should accept the IdP's sign-in directly, map the IdP's groups to its two roles, and keep its password
accounts as a way in when the IdP is down.

## Goals

1. **Generic OpenID Connect**, off by default: any IdP with discovery (Pocket ID, Keycloak, Authentik, …),
   configured under `auth.oidc` in `config.yaml`, with the client secret in `secrets/`.
2. **Authorization Code flow with PKCE (S256)**, `state` and `nonce`, for confidential and public clients.
3. **Full ID token validation**: signature against the IdP's JWKS (cached, rotation-safe), `iss`, `aud`/`azp`,
   `exp`, `iat`, `nonce`; asymmetric algorithms only.
4. **Roles from a claim at every sign-in.** Values map to `admin` or `viewer`; no match means no access, and a
   removed group ends access at the next sign-in (and after `reauth_after_s` at the latest).
5. **Auto-provisioning** keyed by issuer + `sub`, never by username or email.
6. **The same session** as a password sign-in: cookie flags, TTL, CSRF token, audit.
7. **Login page**: a button with a configurable label, an optional auto-redirect, readable errors.
8. **Break-glass**: password sign-in stays on by default; it can only be turned off while OIDC is on.

## Non-goals

- SAML, LDAP, SCIM, or more than one IdP at a time.
- Linking an SSO identity to an existing password account (an account-takeover path).
- RP-initiated or back-channel logout: **Log out** ends the DGXMM session only.
- Refresh tokens: nothing is kept from the IdP after the sign-in.
- Trusting identity headers from a reverse proxy.

## Design

**Flow.** `GET /api/auth/oidc/login?return_to=…` checks the IdP's discovery document, stores a pending flow
(in memory, keyed by the SHA-256 of a 256-bit `state`, holding the nonce, the PKCE verifier, the redirect URI,
the sanitised `return_to` and the SHA-256 of a browser-binding cookie), sets that cookie (`SameSite=Lax`, so it
survives the IdP's redirect) and redirects to the authorization endpoint. The callback pops the flow (single
use), checks the binding cookie and RFC 9207's `iss`, exchanges the code with the verifier, validates the ID
token, reads the role claim (falling back to userinfo when the ID token lacks it), provisions or updates the
user, and starts a normal session. Every failure is a 303 to `/login?sso_error=<code>`.

**Why in memory.** A pending flow lives for 10 minutes; losing it on a restart costs one retry, while storing
verifiers on disk would only add something to protect. DGXMM is a single process, like its rate limits.

**Data.** `users.auth_source` (`sso`), `sso_identities` (issuer, subject, email; unique issuer + subject;
cascades with the user), `sessions.created_at` (for `reauth_after_s`). All are added at start without a
versioned migration (a new table and nullable columns), and older versions ignore them.

**Secrets.** `secrets/oidc_client_secret` is a compose secret (the installer creates it empty), read at each
code exchange so it can be rotated without a restart. No token, code, state, nonce or verifier is logged or
audited, and the callback's query string is dropped from the access log.

## Anti-goals (failure even if it "works")

- A sign-in that succeeds for someone in no mapped group.
- A redirect after sign-in to anywhere but DGXMM's own UI.
- A lock-out: an IdP outage must leave the local admin a way in.
- A code, token or secret in a log line.

## Tests

`tests/test_oidc.py` runs the flow against a fake IdP (`httpx.MockTransport`): discovery and its checks, JWKS
caching, rotation and throttling, valid, expired, wrong-audience, wrong-issuer, wrong-nonce, too old,
future-dated, unsigned, HMAC and wrong-key tokens, PKCE (the fake IdP verifies S256), state replay, browser
binding, the `iss` parameter, role mapping and denial, session end after a lost group, `return_to`,
auto-provisioning, usernames, rate limits and the access-log filter. `frontend/src/pages/Login.test.tsx`
covers the button, the auto-redirect and when it must not happen.
