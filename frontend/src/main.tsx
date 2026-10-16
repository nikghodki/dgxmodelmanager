import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import { AuthProvider, RequireAuth } from "./auth";
import { AppShell } from "@/components/app/app-shell";
import { ChangePassword } from "./pages/ChangePassword";
import { Dashboard } from "./pages/Dashboard";
import { Deployments } from "./pages/Deployments";
import { DeployWizard } from "./pages/DeployWizard";
import { Downloads } from "./pages/Downloads";
import { Login } from "./pages/Login";
import { Models } from "./pages/Models";
import { Users } from "./pages/Users";
import { ThemeProvider, useTheme } from "@/components/app/theme";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import { PageSkeleton } from "@/components/app/states";

const Catalog = lazy(() => import("./pages/Catalog").then((m) => ({ default: m.Catalog })));
const Planner = lazy(() => import("./pages/Planner").then((m) => ({ default: m.Planner })));
const DeploymentDetail = lazy(() => import("./pages/DeploymentDetail").then((m) => ({ default: m.DeploymentDetail })));
const Audit = lazy(() => import("./pages/Audit").then((m) => ({ default: m.Audit })));
const lazyEl = (el: React.ReactNode) => <Suspense fallback={<PageSkeleton />}>{el}</Suspense>;

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });

function ThemedToaster() {
  const { resolved } = useTheme();
  return <Toaster theme={resolved} position="bottom-right" richColors />;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <ThemeProvider>
        <TooltipProvider delayDuration={200}>
          <BrowserRouter basename="/dgxmm">
            <AuthProvider>
              <Routes>
                <Route path="/login" element={<Login />} />
                <Route element={<RequireAuth><AppShell /></RequireAuth>}>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/deployments" element={<Deployments />} />
                  <Route path="/deployments/:id" element={lazyEl(<DeploymentDetail />)} />
                  <Route path="/deploy" element={<DeployWizard />} />
                  <Route path="/models" element={<Models />} />
                  <Route path="/catalog" element={lazyEl(<Catalog />)} />
                  <Route path="/downloads" element={<Downloads />} />
                  <Route path="/planner" element={lazyEl(<Planner />)} />
                  <Route path="/users" element={<Users />} />
                  <Route path="/audit" element={lazyEl(<Audit />)} />
                  <Route path="/change-password" element={<ChangePassword />} />
                </Route>
              </Routes>
            </AuthProvider>
          </BrowserRouter>
          <ThemedToaster />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
