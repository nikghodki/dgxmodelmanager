---
description: "Models on disk, deleting weights, cleaning up downloads, and the Hugging Face catalog with fit badges and the compatibility check."
---

# Models and catalog

## Models on disk

![Models on disk with kinds, formats, sizes and the Deploy and Delete actions](../images/models-light.png#only-light) ![Models on disk with kinds, formats, sizes and the Deploy and Delete actions](../images/models-dark.png#only-dark)

**On disk** lists the models DGXMM found in the Hugging Face cache and the GGUF directories: format,
kind, quantization, size, which engines can serve them, and whether the download is complete
(architecture and context length are shown on wide screens). Filter by kind: **All**, **LLM**,
**Draft** (speculative-decoding draft models such as EAGLE or Medusa) or **Other**. The kind comes
from rules (embeddings, rerankers, speech, vision, drafts and decision models such as classifiers are
recognised from their files); when no rule recognises a model and the decision model is on, the kind
has an **AI** marker whose tooltip gives the model's confidence (see [model kind](decision-model.md#model-kind)). **Deploy** opens the wizard with the model selected. **Rescan** (admins) rescans right
away; otherwise DGXMM rescans every 5 minutes. To add directories, see
[configuration](../configuration/models-and-engines.md#model-directories).

### Delete weights

**Delete…** (admins) removes a model from disk:

- a Hugging Face model: the whole repo directory in the cache (`models--org--name`, every revision)
  and its lock directory;
- a GGUF file: the file, and all its shards if it is a split GGUF. A parent directory left empty is
  removed too, unless it is a configured model directory.

The dialog first shows what will go: the space it frees, the host paths, and any deployments that use
the model. Type the model's short name (the last part of its name, for example `Qwen3-0.6B`, or the
`.gguf` file name) to confirm.

- A model used by a deployment that is running, starting, loading, sleeping or being started cannot be
  deleted. The dialog names the deployment: stop it first. The same applies while a download of the
  model is queued or running.
- Stopped or failed deployments that use the model are listed as a warning. You can still delete it,
  but those deployments will fail to start until the model is back on disk.
- When another revision of the same repo shares a file, only the link is removed and the dialog says
  that no space is freed for it.
- Newer Hugging Face downloads keep file contents in a shared store in the cache (`hub/blobs`), so that
  identical files are stored once across models. Deleting a model also removes the shared files that
  only that model uses; files another model also uses are kept, and the dialog says how many. The space
  shown counts every file once. If a download reuses one of those files while the deletion runs, that
  file is kept and the notification says so.

The table refreshes when the deletion is done, and a notification shows the space freed. Files owned by
root (left behind by engine containers) are deleted too; see
[how deletion works](../architecture/storage.md#deleting-weights).

### Clean up incomplete downloads

Interrupted downloads and models with missing files are marked **Incomplete**. For those rows, admins get
**Clean up** instead of Deploy. When there are any, the header shows **Clean up incomplete (N · size)**,
which lists them all with checkboxes (all selected). Items that cannot be cleaned up right now are
disabled with the reason: a download is running, a deployment uses the model, or a partial file was
written in the last 10 minutes. The same rules apply to a row's **Clean up**: its dialog shows the reason and
keeps the confirm button disabled. Clean up only ever deletes incomplete models, never complete ones.

## Catalog and downloads

![Catalog search with the details sheet of a model: compatibility line, memory estimate and other variants](../images/catalog-light.png#only-light) ![Catalog search with the details sheet of a model: compatibility line, memory estimate and other variants](../images/catalog-dark.png#only-dark)

**Catalog** searches Hugging Face. Turn on **GGUF (llama.cpp)** to search for GGUF repos, and set the
**Context** the fit badges should assume. Each result shows downloads, approximate size, a **fit**
badge based on memory free right now (*Fits now*, *Fits if you evict* running deployments, *Too big*),
whether vLLM supports the architecture (✓, ✗, or ? when unknown) or llama.cpp applies, and llmfit's
assessment (a ⚠ means llmfit's memory figure differs from DGXMM's by over 20 %). Badges show when a
model is already on disk, gated, or a known quantization.

Open a result for details: the [compatibility line](deploying.md#compatibility-check) for vLLM and llama.cpp, the
DGXMM memory estimate at your context length, then **Download safetensors** or a single file from the
list of **GGUF files**. Other quantizations of the same model are
listed too. Downloads are refused if they would leave less than `min_free_disk_gib` (50 GiB) free.
Gated models need an [HF token](../configuration/models-and-engines.md#hugging-face-token).

**Downloads** shows progress, and lets admins cancel. Cancel stops the file being downloaded at once
and removes its partial file; files that already finished are kept. A download interrupted by a DGXMM
restart is marked failed. Start it again: files that already finished are kept. Until every file of a
download is on disk, the model is listed as **Incomplete** (it is not offered for deployment), so a
cancelled or failed download can be resumed or removed with **Clean up**.

## Next steps

- [Deploying a model](deploying.md)
- [Downloads and weight deletion (architecture)](../architecture/storage.md)
