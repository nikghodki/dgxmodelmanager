export function passwordProblems(next: string, again: string, current?: string): string[] {
  const out: string[] = [];
  if (next.length < 12) out.push("Use at least 12 characters");
  if (again && next !== again) out.push("Passwords do not match");
  if (current && next && next === current) out.push("Choose a password different from the current one");
  return out;
}
