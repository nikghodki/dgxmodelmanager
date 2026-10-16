import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import { AuthProvider, RequireAuth } from "./auth";
import { Layout } from "./components/Layout";
import { ChangePassword } from "./pages/ChangePassword";
import { Dashboard } from "./pages/Dashboard";
import { DeploymentDetail } from "./pages/DeploymentDetail";
import { DeployWizard } from "./pages/DeployWizard";
import { Login } from "./pages/Login";
import { Models } from "./pages/Models";
import { Users } from "./pages/Users";

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <BrowserRouter basename="/dgxmm">
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/change-password" element={<RequireAuth><ChangePassword /></RequireAuth>} />
            <Route element={<RequireAuth><Layout /></RequireAuth>}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/models" element={<Models />} />
              <Route path="/deploy" element={<DeployWizard />} />
              <Route path="/deployments/:id" element={<DeploymentDetail />} />
              <Route path="/users" element={<Users />} />
            </Route>
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
