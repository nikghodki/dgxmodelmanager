import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import { AuthProvider, RequireAuth } from "./auth";
import { AppShell } from "@/components/app/app-shell";
import { Audit } from "./pages/Audit";
import { Catalog } from "./pages/Catalog";
import { ChangePassword } from "./pages/ChangePassword";
import { Dashboard } from "./pages/Dashboard";
import { DeploymentDetail } from "./pages/DeploymentDetail";
import { Deployments } from "./pages/Deployments";
import { DeployWizard } from "./pages/DeployWizard";
import { Downloads } from "./pages/Downloads";
import { Login } from "./pages/Login";
import { Models } from "./pages/Models";
import { Planner } from "./pages/Planner";
import { Users } from "./pages/Users";
import { ThemeProvider, useTheme } from "@/components/app/theme";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";

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
                  <Route path="/models" element={<Models />} />
                  <Route path="/catalog" element={<Catalog />} />
                  <Route path="/downloads" element={<Downloads />} />
                  <Route path="/planner" element={<Planner />} />
                  <Route path="/deploy" element={<DeployWizard />} />
                  <Route path="/deployments/:id" element={<DeploymentDetail />} />
                  <Route path="/users" element={<Users />} />
                  <Route path="/audit" element={<Audit />} />
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
