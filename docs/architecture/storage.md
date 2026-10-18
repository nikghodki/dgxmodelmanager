---
description: "How downloads run and how weights are deleted safely, including the shared blob store and the delete helper."
---

# Downloads and weight deletion

## Downloads

`downloads.py` fetches the selected files one at a time. Each file is fetched by `hf_hub_download` in a
child process (`python -I -B`, progress bars off; the HF token is passed in its environment, never on its
command line). Cancel kills the child's process group, which aborts the HTTP stream at once, then removes
the repo's `blobs/*.incomplete` partial files (only one download per repo runs at a time) and rescans the
inventory. When the download starts it writes `.dgxmm-download.json` in the repo dir with the revision and
the file list, and removes it only once every file is on disk. The inventory reports a snapshot of that
revision that lacks a listed file as incomplete ("download not finished"), so a cancelled or failed
download is never offered as a complete model; incomplete cleanup offers it instead. A marker for another
revision, or an unreadable one, is ignored.

A supported decision model (d3-mini, d3-nano) never takes that path, whether it is started from the Decision
model page or `POST /api/downloads`: `DownloadService.start_pinned` runs `decider/snapshot.py` (the installer's
and the CLI's downloader) in a daemon worker thread. It fetches only the pinned revision's files named by the
checked-in hash list and the verified manifest, hashes each while streaming and links it only when it matches,
resumes a partial file with HTTP Range on the next attempt, and never sends the HF token. A snapshot that is
already complete is only verified. Progress (the planned files and their sizes, then the bytes on disk) goes to
the same Download row and `download` events; Cancel stops the worker before its next chunk. Another revision is
refused (`pinned_revision`).

## Deleting weights

`inventory/weights.py` plans and performs deletions; the API (`DELETE /api/inventory?ref=`, with
`GET /api/inventory/delete-preview` and `/api/inventory/cleanup-incomplete[/preview]`) is admin-only.

**Plan.** A Hugging Face ref maps to `<hub>/models--org--name` plus its `.locks` directory. A GGUF path
maps to the file and all its split shards. When a GGUF inside the HF cache is a snapshot symlink, the
blob it points to is deleted too, unless another snapshot link still uses that blob (then only the link
goes, and 0 bytes are reported).

**Shared blob store (huggingface_hub >= 2).** A repo blob may itself be a symlink into the cache-wide
store `<hub>/blobs/<xx>/<hash>`, which holds each distinct file once for all repos (with a
`<hash>.refs` manifest and a `<hash>.lock`). The plan collects the store entries that the deleted links
resolve to and then, read-only and before anything is removed, scans every symlink in the cache (all
`models--`, `datasets--` and `spaces--` repos) and in the GGUF roots outside it. Entries another link
still resolves to are kept; if part of the tree cannot be read, no store entry is removed. Only
well-formed entries count (a regular file `<xx>/<64 hex>` whose real path stays inside the store), and
the store directory itself is a protected root. Freed bytes count every real file once: repo files
that are not links, plus the exclusive store entries (0 for an entry with another hard link). Store
entries are removed after the repo's own files, one at a time under the entry's `.lock` (the lock
huggingface_hub holds while it links an entry into a repo and records the link in the manifest). Under
the lock the manifest is re-read, so an entry that a concurrent download has just reused is kept, and
the result says so. A `.lock` held for over 10 s also keeps the entry. Removing an entry also removes
its manifest; root-owned entries go through the helper while the lock is held. Incomplete-download
cleanup uses the same plan.

**Safety rules**, enforced on the server:

- **Containment.** Refs containing `..` are refused. Both the link location and the fully resolved path
  must lie inside the HF hub or a configured GGUF root, so symlinks cannot escape. A target that is, or
  contains, a configured root (hub, `hf_cache.local`, `.locks`, the shared store `blobs`, a GGUF root) is refused (`400
  unsafe_path`). Symlink loops and other path errors are refused the same way.
- **In use.** Every deployment's model and draft model are mapped to paths. A non-terminal deployment
  whose paths lie in a target blocks the delete (`409 model_in_use`, listing the deployments). Stopped
  and failed ones are returned as warnings.
- **Downloads.** A queued or running download of the repo blocks it (`409 download_active`).
- **Races.** The deleter publishes the targets it is about to delete, then plans again with a fresh
  database read. Deploy, start, restart, switch, reconfigure and download starts check those markers
  (and check again after creating their row), so either the deleter sees the new deployment or the
  deployment sees the deletion and is refused (`409 model_being_deleted`). Deletions are serialised.

**Execution.** One helper container runs per parent directory, from DGXMM's own image (`helper_image`;
it is never pulled):

- user `0:0`, so files left owned by root by engine containers can be removed;
- one bind mount: the target's parent host directory at `/x`, and nothing else;
- command `rm -rf -- /x/<name>…` (`rm -f` for files) as the entrypoint;
- `network_mode: none`, read-only root filesystem, all capabilities dropped except `DAC_OVERRIDE` and
  `FOWNER`, `no-new-privileges`, a process limit, `auto_remove`;
- label `dgxmm.helper=true` (never `dgxmm.managed`), and a `helper_timeout_s` (120 s) limit after which
  only that container is force-removed.

Anything the helper could not remove, or everything when the helper cannot run (no Docker, image
missing), is deleted in-process. A permission error there fails with `500 delete_failed` and a
`sudo chown -R <uid>:<gid> <path>` hint. Afterwards the inventory is refreshed. Every deletion, refusal
and failure is audited (`model_delete`, `model_delete_refused`, `model_delete_failed`; clean-ups add one
`incomplete_cleanup` summary row).

**Incomplete clean-up** applies the same rules to artifacts that are not complete only. Complete models
are never deleted through it, and Hugging Face repos with a partial blob written in the last 10 minutes
are skipped as `recently_active`.

## Next steps

- [Models and catalog (user guide)](../user-guide/models.md)
- [Process view](processes.md)
