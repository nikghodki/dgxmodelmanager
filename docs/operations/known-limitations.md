---
description: "What DGXMM does not do, or does only on a best-effort basis."
---

# Known limitations

- **Command-line masking is best-effort.** The Processes page and the dashboard's external memory card
  mask passwords, tokens and keys, and mask a whole value when it cannot be parsed reliably (fail
  closed). Some rare wrapper or quoting forms may still not be recognised, so a secret passed as a
  command-line argument can appear unmasked. Do not pass secrets as arguments; use files or
  environment variables, which DGXMM never reads.
- **Deletion only knows about DGXMM's deployments.** A model is protected from deletion when a DGXMM
  deployment uses it. Containers or processes that DGXMM does not manage (a vLLM started by hand, a
  notebook) and use the same HF cache or GGUF directory are not detected. Check the
  [Processes](../user-guide/processes.md) page, or `docker ps`, before you delete a model.
- **Clearing the master key does not deregister models.** Only turning the integration off or changing
  the internal URL removes DGXMM's models from LiteLLM. Clearing a master key stored on the LiteLLM
  page (or removing the secret file) leaves the existing registrations in LiteLLM; turn the
  integration off first if you want them removed.
- **Virtual-key limits cannot be cleared.** LiteLLM's key update cannot remove a budget or rate limit once it is
  set, so **Edit** can only change it. Set a higher value, or create a new key
  and revoke the old one.
- **Per-process GPU memory on GB10.** Per-process GPU figures come from NVML. When NVML reports no value
  for a process, DGXMM shows `0`, so the GPU column on the Processes page may read 0 for a process that
  does use the GPU. On the GB10's unified memory, the RAM column and the deployment's measured memory
  are the more reliable figures.

## Next steps

- [Security hardening](../security/hardening.md)
