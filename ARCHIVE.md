# Archive handoff

## Current state

The archive-readiness branch consolidates the maintained engine around one
neutral public API and fixes the P0 runtime defects found during review:

- face landmarks are batched per frame and per face;
- frame dimensions and flattened multi-face offsets are preserved;
- multi-person pose results have deterministic ownership and refresh state;
- Unity, WebAssembly, and face-geometry C APIs validate inputs and expose
  matching destroy functions;
- success is consistently represented by status code zero;
- native build targets use portable install destinations and centralized
  dependency paths;
- examples, standalone tools, and experimental 6D tracking are disabled by
  default;
- generated build products and explicitly deprecated prototypes are no longer
  tracked;
- company-specific identifiers and workstation paths have been removed.

## Breaking surface change

All maintained public names now use `Custom`, `CUSTOM`, or `custom` according
to their existing case convention. This changes native library names, C/C++
symbols, WebAssembly bindings, namespaces, status types, and examples.

Downstream consumers must update their imports and native declarations before
adopting the archive branch. No compatibility aliases are retained because the
goal is to remove the former-company surface completely.

## Validation completed

- Python build driver byte-compilation
- CMake preset JSON validation
- dependency-free structural regression check
- exhaustive scan of maintained text files for retired identifiers and
  developer-home paths
- fast-forward-only branch publishing for every isolated change

The current workspace does not contain CMake, OpenCV, ONNX Runtime, Emscripten,
Android NDK, Xcode, or Visual Studio. A full platform compilation was therefore
not performed here and must not be inferred from the structural validation.

## Remaining archive decisions

Before making a public archive, the owner should decide whether to:

1. add an explicit project license;
2. publish checksums and provenance for the prebuilt dependency archives;
3. move demo models and large media to release assets or Git LFS;
4. run at least one native smoke build on each platform that must remain
   supportable;
5. tag the final merge commit and publish a short immutable release note.

Experimental 6D tracking remains available behind
`AI_ENGINE_BUILD_6D_TRACKING=ON`, but it is not part of the default archive
build and requires additional graphics dependencies.

## Final GitHub checklist

1. Review and merge the archive-readiness pull request.
2. Run `python3 scripts/check_archive_readiness.py` from the merge commit.
3. Complete the licensing and binary-provenance decisions above.
4. Create an archive tag or release if a stable reference is desired.
5. Use GitHub repository settings to archive the repository.
