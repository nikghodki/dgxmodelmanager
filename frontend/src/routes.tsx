import React, { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { RequireAuth } from "./auth";
import { AppShell } from "@/components/app/app-shell";
import { PageSkeleton } from "@/components/app/states";
import { ChangePassword } from "./pages/ChangePassword";
import { Dashboard } from "./pages/Dashboard";
import { Deployments } from "./pages/Deployments";
import { Downloads } from "./pages/Downloads";
import { Login } from "./pages/Login";
import { Models } from "./pages/Models";
import { Users } from "./pages/Users";

const DeployWizard = lazy(() => import("./pages/DeployWizard").then((m) => ({ default: m.DeployWizard })));
const Catalog = lazy(() => import("./pages/Catalog").then((m) => ({ default: m.Catalog })));
const Planner = lazy(() => import("./pages/Planner").then((m) => ({ default: m.Planner })));
const DeploymentDetail = lazy(() => import("./pages/DeploymentDetail").then((m) => ({ default: m.DeploymentDetail })));
const LiteLLM = lazy(() => import("./pages/LiteLLM").then((m) => ({ default: m.LiteLLM })));
const Processes = lazy(() => import("./pages/Processes").then((m) => ({ default: m.Processes })));
const DecisionModel = lazy(() => import("./pages/DecisionModel").then((m) => ({ default: m.DecisionModel })));
const Audit = lazy(() => import("./pages/Audit").then((m) => ({ default: m.Audit })));
const lazyEl = (el: React.ReactNode) => <Suspense fallback={<PageSkeleton />}>{el}</Suspense>;

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<RequireAuth><AppShell /></RequireAuth>}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/deployments" element={<Deployments />} />
        <Route path="/deployments/:id" element={lazyEl(<DeploymentDetail />)} />
        <Route path="/deploy" element={lazyEl(<DeployWizard />)} />
        <Route path="/models" element={<Models />} />
        <Route path="/catalog" element={lazyEl(<Catalog />)} />
        <Route path="/downloads" element={<Downloads />} />
        <Route path="/planner" element={lazyEl(<Planner />)} />
        <Route path="/processes" element={lazyEl(<Processes />)} />
        <Route path="/users" element={<Users />} />
        <Route path="/audit" element={lazyEl(<Audit />)} />
        <Route path="/litellm" element={lazyEl(<LiteLLM />)} />
        <Route path="/decision-model" element={lazyEl(<DecisionModel />)} />
      </Route>
      {/* Outside the shell: must-change-password users land here first and get the plain auth layout. */}
      <Route path="/change-password" element={<RequireAuth><ChangePassword /></RequireAuth>} />
    </Routes>
  );
}
