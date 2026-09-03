ACCEPT.

Reviewed the diff in the worktree. Both files match the contract exactly:

- `f_home.xml`: `rcVideoList` is wrapped in `SwipeRefreshLayout` id `swipeRefresh`; the four constraint attributes and 0dp sizing moved onto the wrapper; RecyclerView keeps its id and every other attribute, now `match_parent`/`match_parent`.
- `HomeFragment.kt`: `setOnRefreshListener` calls `viewModel.downloadData(force = false)`; `isRefreshing = false` set in `LoadState.Error`, `LoadState.NotLoading`, and `onPause()`; `pbLoading` visibility now also checks `!swipeRefresh.isRefreshing` so the two spinners don't overlap.

No scope violations, no unrelated changes, `btnUpdateList`/`HomeViewModel`/DI untouched as required. Compile passed for real (`:app:compileDebugKotlin`, exit 0) in the isolated worktree, not just claimed by the model.

Merging: `git -C "<repo>" diff <worktree_path> -- <files>` applied to the real working tree, or simply `git worktree` the same branch and cherry-pick the two file changes, then `local_cleanup` the worktree.
