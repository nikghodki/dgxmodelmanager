import { describe, expect, it } from "vitest";
import { titleFor, visibleGroups } from "@/components/app/nav";

describe("nav", () => {
  it("hides the Admin group from viewers", () => {
    expect(visibleGroups(false).map((g) => g.label)).toEqual(["Operate", "Models"]);
    expect(visibleGroups(true).map((g) => g.label)).toEqual(["Operate", "Models", "Admin"]);
    expect(visibleGroups(false).flatMap((g) => g.items.map((i) => i.to))).not.toContain("/users");
  });
  it.each([
    ["/", { section: "Dashboard" }],
    ["/deployments", { section: "Deployments" }],
    ["/deployments/7", { section: "Deployments", item: "Deployment" }],
    ["/deploy", { section: "Deployments", item: "Deploy a model" }],
    ["/catalog", { section: "Catalog" }],
    ["/models", { section: "On disk" }],
    ["/change-password", { section: "Account", item: "Change password" }],
    ["/nope", { section: "DGXMM" }],
  ])("titleFor(%s)", (path, want) => expect(titleFor(path)).toEqual(want));
});
