/** Single sign-on helpers for the login page (backend: dgxmm/api/sso.py). */

/** Readable text for the `?sso_error=<code>` the callback redirects to. */
const MESSAGES: Record<string, string> = {
  denied: "Your account is not in a group that may use DGXMM. Ask an admin to add you, then sign in again.",
  not_provisioned: "Your account has no DGXMM user yet. Ask an admin to let it sign in.",
  idp_unavailable: "The sign-in service could not be reached. Try again in a moment.",
  idp_error: "The sign-in service refused the request. Try again; if it keeps failing, an admin can find the reason in the DGXMM log.",
  idp_denied: "Sign-in was cancelled or refused at the sign-in service.",
  invalid_state: "That sign-in expired or was already used. Start it again.",
  invalid_token: "The sign-in could not be verified. Try again; if it keeps failing, an admin can find the reason in the DGXMM log.",
  rate_limited: "Too many sign-in attempts. Wait a minute and try again.",
  busy: "Too many sign-ins are in progress. Try again in a few minutes.",
  sso_disabled: "Single sign-on is not turned on.",
};

export function ssoErrorMessage(code: string | null | undefined): string | null {
  if (!code) return null;
  return MESSAGES[code] ?? "Single sign-on failed. Try again.";
}

/** The URL that starts a sign-in: <base><login_path>?return_to=<UI path>. Only UI paths are passed (the server
 * re-checks them); the login page itself is never a target. */
export function ssoLoginHref(base: string, loginPath: string, from: string | undefined): string {
  const target = from && from.startsWith("/") && !from.startsWith("//") && from !== "/login" ? from : "/";
  return `${base}${loginPath}?return_to=${encodeURIComponent(target)}`;
}
