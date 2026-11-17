---
description: "Single sign-on with OpenID Connect: Pocket ID, Keycloak, Authentik; roles from groups, auto-redirect and the break-glass password login."
---

# Single sign-on (OpenID Connect)

DGXMM can let people sign in with an OpenID Connect identity provider (IdP) such as Pocket ID, Keycloak or
Authentik, next to (or instead of) its own password accounts. It is off until you turn it on in
`config.yaml`.

- The sign-in is the Authorization Code flow with PKCE (S256), so it works for confidential clients (with a
  client secret) and public ones (PKCE only).
- The DGXMM role comes from a claim of the ID token, usually `groups`, at **every** sign-in. Someone in none
  of the configured groups is refused, and removing a group at the IdP removes access at their next sign-in.
- A successful sign-in creates an ordinary DGXMM session: the same cookie, lifetime and CSRF protection as a
  password sign-in.
- Password accounts, the first-run `admin` and the admin CLI keep working, so the local `admin` stays your
  break-glass login.

## Set it up

1. **Register DGXMM at your IdP** as an OpenID Connect client:
   - Redirect (callback) URI: `https://dgx.example.com/dgxmm/api/auth/oidc/callback`, with your own host.
   - Grant: authorization code, with PKCE S256.
   - Scopes: `openid email profile groups`, or whatever makes your IdP put the groups into the ID token.
   - Allowed users: restrict the client to the groups that should reach DGXMM, if your IdP can (Pocket ID:
     allowed user groups).
2. **Store the client secret**, if the IdP issued one, in `secrets/oidc_client_secret` next to
   `docker-compose.yaml`. The installer creates that file empty, with mode 600. Write the secret into it
   without echoing it, for example by pasting it at the prompt:

   ```bash
   read -rs SECRET && printf '%s' "$SECRET" > secrets/oidc_client_secret && unset SECRET
   ```

   DGXMM reads the file at every sign-in, so rotating the secret needs no restart. Leave it empty for a public
   client.
3. **Configure `config.yaml`**:

   ```yaml
   auth:
     oidc:
       enabled: true
       issuer: https://id.example.com          # exactly as in the IdP's discovery document
       client_id: dgxmm
       public_url: https://dgx.example.com     # how browsers reach DGXMM (origin only)
       button_label: Sign in with Pocket ID
       role_mapping:
         claim: groups
         admin: [dgxmm-admins]
         viewer: [dgxmm-viewers]
       auto_redirect: false                    # true: skip the login page
   ```

4. **Restart DGXMM** (`./install.sh` or `docker compose up -d`). Configuration errors (an `http://` issuer, no
   group in `role_mapping`, a missing `client_id`) stop DGXMM at start, with the reason in
   `./install.sh logs`.
5. Open the login page and click the button. The first sign-in creates the DGXMM user.

### Reaching the IdP from the container

DGXMM fetches the IdP's discovery document, signing keys and tokens from inside its container. Check that it
can:

```bash
docker exec dgxmm python -c "import httpx; print(httpx.get('https://id.example.com/.well-known/openid-configuration').status_code)"
```

- **A private CA.** If the IdP's certificate comes from your own CA, put the CA certificate (PEM) under
  `data/`, which is mounted at `/app/data`, and point `ca_bundle` at it, for example
  `ca_bundle: /app/data/certs/idp-ca.pem`. It replaces the system CA store for calls to the IdP only.
- **A name only your LAN resolves.** If Docker's DNS cannot resolve the IdP's host name, add it in a
  `docker-compose.override.yaml` next to `docker-compose.yaml`. It is untracked and `./install.sh` keeps it:

  ```yaml
  services:
    dgxmm:
      extra_hosts:
        - "id.example.com:192.0.2.10"
  ```

## Options

All of them live under `auth.oidc` in `config.yaml`.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Turn single sign-on on. |
| `issuer` | | The IdP's issuer URL, `https://` only. Discovery reads `<issuer>/.well-known/openid-configuration`, which must name exactly this issuer. |
| `client_id` | | The client id registered at the IdP. |
| `client_secret_file` | `/run/secrets/oidc_client_secret` | The client secret, from `secrets/oidc_client_secret`. Missing or empty: a public client, which sends PKCE only. |
| `redirect_path` | `/api/auth/oidc/callback` | The callback path under the base path. The redirect URI is `<public_url>/dgxmm<redirect_path>`. |
| `public_url` | empty | The origin browsers use, e.g. `https://dgx.example.com`. Empty: the request's scheme (from `X-Forwarded-Proto` when it comes from a [trusted proxy](../configuration/index.md#web-ui-and-sessions)) and its `Host` header. Set it behind a reverse proxy. |
| `scopes` | `[openid, email, profile, groups]` | Scopes to request. Must include `openid`. |
| `ca_bundle` | none | PEM file that replaces the system CA store for calls to the IdP. |
| `button_label` | `Sign in with SSO` | Text of the button on the login page. |
| `role_mapping.claim` | `groups` | The claim with the groups or roles. A dotted path reads a nested claim, e.g. `realm_access.roles`. A list or a single string. If the ID token lacks it, DGXMM asks the IdP's userinfo endpoint. |
| `role_mapping.admin` | `[]` | Claim values that make an admin. |
| `role_mapping.viewer` | `[]` | Claim values that make a viewer. Someone in both lists is an admin, and someone in neither is refused. At least one of the two lists must have a value. |
| `auto_provision` | `true` | Create the DGXMM user at the first sign-in. With `false`, only SSO users that already exist (created earlier) can sign in. |
| `auto_redirect` | `false` | The login page goes straight to the IdP. Open `/dgxmm/login?local=1` for the password form. |
| `local_login` | `true` | Password sign-in. `false` refuses it, even for the local `admin`, so keep it `true` unless you have another way in. It is always on while `enabled` is `false`. |
| `timeout_s` | `10` | Timeout of each call to the IdP. |
| `reauth_after_s` | `43200` (12 h) | An SSO session ends this long after its sign-in, however active it is. The next request then needs a new sign-in, which re-reads the groups. |

## Users and roles

- **Identity.** An SSO user is identified by the IdP's issuer and the token's `sub`. It is never matched to an
  existing DGXMM account by username or email, so no password account can be taken over through the IdP.
- **Username.** Taken from `preferred_username`, else the part of the email before the `@`, else `name`,
  cut down to `a-z 0-9 . - _`. If a local account already has that name, a suffix is added: `alice-2`.
- **Roles.** The role is set at every sign-in from `role_mapping`. On the **Users** page, SSO users have an
  **SSO** badge and their email. Their role can't be changed there and they have no password to reset:
  change their groups at the IdP. Deleting an SSO user ends their sessions. They get a new user at their
  next sign-in if they are still in a mapped group, so to keep someone out, remove the group at the IdP.
- **Your existing admin.** To keep admin rights when you move to SSO, put your IdP account in an `admin`
  group. Your first SSO sign-in then creates a second, SSO admin user. Keep the local `admin` with a strong
  password as break-glass.
- **Account menu.** SSO users see *Single sign-on* and their email in the account menu, without
  *Change password*.

## Auto-redirect and signing out

With `auto_redirect: true`, the login page sends you straight to the IdP. If you already have a session
there, you land back in DGXMM without seeing a login page. The login page does not redirect:

- after **Log out** in that tab: it shows *You are signed out of DGXMM*, with the button to sign in again;
- after a failed sign-in: it shows the reason;
- with `?local=1` in the URL: it shows the password form.

**Log out** ends the DGXMM session only, not your session at the IdP.

## Behind an authenticating reverse proxy

If a sign-in proxy (oauth2-proxy, Authelia, a forward-auth gateway) already guards DGXMM with the same IdP,
DGXMM's own sign-in completes without a prompt, because the IdP still has the session. Register DGXMM as its
own client: it validates its own ID tokens and does not trust identity headers set by the proxy.

## When sign-in fails

The login page shows a short reason, and the DGXMM log (`./install.sh logs`) has the details. Codes,
tokens and secrets are never logged.

| Message | Likely cause |
|---|---|
| *not in a group that may use DGXMM* | The account has none of the `role_mapping` values in the claim. Check the group at the IdP, and that the scopes make the IdP include the claim (Pocket ID: `groups`). |
| *could not be reached* | DGXMM can't reach the IdP: DNS, firewall, or the certificate is not trusted (`ca_bundle`). |
| *refused the request* | A wrong client id or secret, or the redirect URI is not registered at the IdP (compare it with `public_url`). |
| *expired or was already used* | The sign-in took longer than 10 minutes, was started in another browser, DGXMM restarted in between, or the page was reloaded. Start again. |
| *could not be verified* | The ID token failed a check: clock skew over 60 s, a wrong `issuer`, or another audience. |
| *no DGXMM user yet* | `auto_provision: false` and this identity never signed in before. |
| *Too many sign-in attempts* | More than 20 sign-in steps (starts and callbacks) per minute from one address. |

## Examples

=== "Pocket ID"

    In Pocket ID: **OIDC Clients → Add**, callback `https://dgx.example.com/dgxmm/api/auth/oidc/callback`, PKCE
    on, allowed user groups `dgxmm-admins` (and `dgxmm-viewers`). Copy the client secret into
    `secrets/oidc_client_secret`.

    ```yaml
    auth:
      oidc:
        enabled: true
        issuer: https://id.example.com
        client_id: 00000000-0000-0000-0000-000000000000   # the id Pocket ID shows
        public_url: https://dgx.example.com
        button_label: Sign in with Pocket ID
        role_mapping: {claim: groups, admin: [dgxmm-admins], viewer: [dgxmm-viewers]}
    ```

    Pocket ID adds the `groups` claim only when the `groups` scope is requested (it is by default).

=== "Keycloak"

    In the realm: **Clients → Create**, OpenID Connect, client authentication on, standard flow, valid
    redirect URI `https://dgx.example.com/dgxmm/api/auth/oidc/callback`; under **Advanced**, PKCE method
    S256. For groups, add a *Group Membership* mapper (token claim name `groups`, full group path off) to
    the client's dedicated scope, or map realm roles instead:

    ```yaml
    auth:
      oidc:
        enabled: true
        issuer: https://sso.example.com/realms/lab
        client_id: dgxmm
        public_url: https://dgx.example.com
        button_label: Sign in with Keycloak
        scopes: [openid, email, profile]
        role_mapping: {claim: realm_access.roles, admin: [dgxmm-admin], viewer: [dgxmm-viewer]}
    ```

=== "Authentik"

    Create an **OAuth2/OpenID Provider** (confidential, redirect URI
    `https://dgx.example.com/dgxmm/api/auth/oidc/callback`, scopes `openid email profile`) and an
    application with slug `dgxmm`. Authentik's `profile` scope includes `groups`. The issuer ends in a slash:

    ```yaml
    auth:
      oidc:
        enabled: true
        issuer: https://authentik.example.com/application/o/dgxmm/
        client_id: dgxmm
        public_url: https://dgx.example.com
        button_label: Sign in with Authentik
        scopes: [openid, email, profile]
        role_mapping: {claim: groups, admin: [DGXMM Admins], viewer: [DGXMM Viewers]}
    ```

## Next steps

- [Users and roles](users.md)
- [Threat model: single sign-on](../security.md#single-sign-on-openid-connect)
- [Networking and reverse proxy](../configuration/networking.md)
