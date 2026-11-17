import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import * as api from "@/api";
import * as auth from "@/auth";
import { goTo, Login } from "./Login";
import type { AuthConfig } from "@/types";

vi.mock("@/auth");
vi.mock("@/api", async (orig) => ({ ...(await orig<typeof import("@/api")>()), api: vi.fn() }));

const SSO: AuthConfig = { local_login: true, oidc: { label: "Sign in with Example ID", auto_redirect: false, login_path: "/api/auth/oidc/login" } };

function setup(config: AuthConfig | Error, { url = "/login", from, loggedOut = false }: { url?: string; from?: string; loggedOut?: boolean } = {}) {
  vi.mocked(auth.useAuth).mockReturnValue({ user: null, loading: false, loggedOut, login: vi.fn(), logout: vi.fn(), reload: vi.fn() });
  vi.mocked(api.api).mockImplementation(async (path: string) => {
    if (path !== "/auth/config") throw new Error(`unexpected ${path}`);
    if (config instanceof Error) throw config;
    return config;
  });
  return render(
    <MemoryRouter initialEntries={[{ pathname: url.split("?")[0], search: url.includes("?") ? "?" + url.split("?")[1] : "", state: from ? { from } : undefined }]}
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Login />
    </MemoryRouter>
  );
}

describe("Login", () => {
  let assign: ReturnType<typeof vi.spyOn>;
  beforeEach(() => { assign = vi.spyOn(goTo, "assign").mockImplementation(() => undefined); });
  afterEach(() => { vi.restoreAllMocks(); });

  it("offers the SSO button and the password form", async () => {
    setup(SSO, { from: "/deployments/3" });
    const link = await screen.findByRole("link", { name: /Sign in with Example ID/ });
    expect(link.getAttribute("href")).toBe(`${api.basePath}/api/auth/oidc/login?return_to=%2Fdeployments%2F3`);
    expect(screen.getByLabelText("Username")).toBeInTheDocument();
    expect(assign).not.toHaveBeenCalled();
  });

  it("redirects straight to the IdP with auto_redirect", async () => {
    setup({ ...SSO, oidc: { ...SSO.oidc!, auto_redirect: true } }, { from: "/models" });
    await waitFor(() => expect(assign).toHaveBeenCalledWith(`${api.basePath}/api/auth/oidc/login?return_to=%2Fmodels`));
    expect(screen.getByRole("status")).toHaveTextContent(/Redirecting/);
  });

  it("does not redirect again after an SSO error, and shows it", async () => {
    setup({ ...SSO, oidc: { ...SSO.oidc!, auto_redirect: true } }, { url: "/login?sso_error=denied" });
    expect(await screen.findByRole("alert")).toHaveTextContent(/not in a group/);
    expect(screen.getByRole("link", { name: /Sign in with Example ID/ })).toBeInTheDocument();
    expect(assign).not.toHaveBeenCalled();
  });

  it("does not redirect right after logging out", async () => {
    setup({ ...SSO, oidc: { ...SSO.oidc!, auto_redirect: true } }, { loggedOut: true });
    expect(await screen.findByText(/signed out/)).toBeInTheDocument();
    expect(assign).not.toHaveBeenCalled();
  });

  it("shows the password form with ?local even when auto_redirect is on", async () => {
    setup({ ...SSO, oidc: { ...SSO.oidc!, auto_redirect: true } }, { url: "/login?local=1" });
    expect(await screen.findByLabelText("Username")).toBeInTheDocument();
    expect(assign).not.toHaveBeenCalled();
  });

  it("hides the password form when local login is off", async () => {
    setup({ ...SSO, local_login: false });
    await screen.findByRole("link", { name: /Sign in with Example ID/ });
    expect(screen.queryByLabelText("Username")).not.toBeInTheDocument();
  });

  it("is a plain password form without SSO", async () => {
    setup({ local_login: true, oidc: null });
    expect(await screen.findByLabelText("Username")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("falls back to the password form when the server has no /auth/config", async () => {
    setup(new Error("404"));
    expect(await screen.findByLabelText("Username")).toBeInTheDocument();
  });
});
