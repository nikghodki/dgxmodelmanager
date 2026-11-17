import { describe, expect, it } from "vitest";
import { ssoErrorMessage, ssoLoginHref } from "./sso";

describe("sso helpers", () => {
  it("builds the sign-in URL with a UI return path", () => {
    expect(ssoLoginHref("/dgxmm", "/api/auth/oidc/login", "/deployments/3")).toBe("/dgxmm/api/auth/oidc/login?return_to=%2Fdeployments%2F3");
    expect(ssoLoginHref("/dgxmm", "/api/auth/oidc/login", undefined)).toBe("/dgxmm/api/auth/oidc/login?return_to=%2F");
    expect(ssoLoginHref("/dgxmm", "/api/auth/oidc/login", "//evil.example.com")).toBe("/dgxmm/api/auth/oidc/login?return_to=%2F");
    expect(ssoLoginHref("/dgxmm", "/api/auth/oidc/login", "/login")).toBe("/dgxmm/api/auth/oidc/login?return_to=%2F");
  });
  it("maps error codes to readable messages", () => {
    expect(ssoErrorMessage("denied")).toMatch(/not in a group/);
    expect(ssoErrorMessage("something-new")).toMatch(/failed/);
    expect(ssoErrorMessage(null)).toBeNull();
  });
});
