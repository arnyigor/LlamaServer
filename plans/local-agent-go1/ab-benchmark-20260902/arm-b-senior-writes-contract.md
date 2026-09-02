The list is re-queried from Room on refresh, and `MainPagingSource` reads only from the local database — so an unchanged database returns an identical list. `downloadData` is the only path that actually fetches new content. Delegating the implementation:

```json
{
  "goal": "Add pull-to-refresh to the home screen that triggers a real data update instead of only re-querying the local database",
  "scope": {
    "allowed_files": [
      "app/src/main/res/layout/f_home.xml",
      "app/src/main/java/com/arny/mobilecinema/presentation/home/HomeFragment.kt"
    ]
  },
  "changes": [
    "In f_home.xml, wrap the existing RecyclerView rcVideoList in an androidx.swiperefreshlayout.widget.SwipeRefreshLayout with id swipeRefresh",
    "Move the RecyclerView's four layout_constraint attributes and its 0dp width/height onto the SwipeRefreshLayout, and give the RecyclerView match_parent width and height",
    "Keep every other RecyclerView attribute and its id unchanged so existing binding references still resolve",
    "In HomeFragment.initListeners, set a setOnRefreshListener on binding.swipeRefresh that calls viewModel.downloadData(force = false)",
    "In the existing addLoadStateListener block, set binding.swipeRefresh.isRefreshing = false in both the LoadState.Error and LoadState.NotLoading branches",
    "In the LoadState.Loading branch, additionally require that swipeRefresh is not already refreshing before showing pbLoading, so the two spinners never appear together",
    "In HomeFragment.onPause, set binding.swipeRefresh.isRefreshing = false"
  ],
  "invariants": [
    "Do not change the RecyclerView id rcVideoList",
    "Do not modify the ConcatAdapter, GridLayoutManager or SpanSizeLookup setup",
    "Do not change what btnUpdateList does",
    "Do not modify HomeViewModel or any file outside allowed_files",
    "Do not add or change any dependency"
  ],
  "acceptance": {
    "compile": [":app:compileDebugKotlin"],
    "search": [
      { "pattern": "SwipeRefreshLayout", "minimum_matches": 1, "path": "app/src/main/res/layout/f_home.xml" },
      { "pattern": "swipeRefresh", "minimum_matches": 4, "path": "app/src/main/java/com/arny/mobilecinema/presentation/home/HomeFragment.kt" }
    ]
  }
}
```
