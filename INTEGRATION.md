# Vulcan integration

Run commands from the Vulcan checkout root so the existing `src` and `scripts` imports resolve correctly. Vulcan is not yet a pip-installable Python project. Use a dedicated environment and the root README setup instructions.

Lillie US tracks Vulcan at `modules/ml_mineral` as a Git submodule. New clones should use `git clone --recurse-submodules` or `git submodule update --init --recursive`. Ordinary submodule initialisation checks out the recorded commit, not a moving branch.

To update Lillie US, first review and test a Vulcan commit, check out that exact commit in its submodule, and update the parent repository gitlink and `source-pins.json` together. Do not advance the pin merely because the upstream main branch changes.

The test-discovery configuration restricts collection to `tests`; dataset-processing helpers in `scripts` are not standalone tests. The 46 supplied tests passed during the earlier source import, which does not establish real geological performance or Southern US validation.

Current manifests record selected source files, not installed dependency versions or every repository document. When changing a file covered by `SOURCE_MANIFEST.json`, update its digest and the documented combined tree hash in the same commit. The archived import manifest records the historical baseline and remains unchanged.
