I'll add pull-to-refresh to the home screen so the list actually fetches fresh data instead of just re-querying the unchanged local database.

## 1. `app/src/main/res/layout/f_home.xml`

Wrap the RecyclerView in a `SwipeRefreshLayout`, moving the RecyclerView's constraints onto the wrapper:

```xml
        <androidx.swiperefreshlayout.widget.SwipeRefreshLayout
            android:id="@+id/swipeRefresh"
            android:layout_width="0dp"
            android:layout_height="0dp"
            app:layout_constraintBottom_toBottomOf="parent"
            app:layout_constraintEnd_toEndOf="parent"
            app:layout_constraintStart_toStartOf="parent"
            app:layout_constraintTop_toBottomOf="@id/homeContentTypeGroup">

            <androidx.recyclerview.widget.RecyclerView
                android:id="@+id/rcVideoList"
                android:layout_width="match_parent"
                android:layout_height="match_parent"
                android:clipToPadding="false"
                android:orientation="vertical"
                android:overScrollMode="never"
                android:paddingHorizontal="4dp"
                android:paddingTop="4dp"
                android:paddingBottom="8dp"
                app:layoutManager="GridLayoutManager"
                app:spanCount="2"
                tools:listitem="@layout/i_home_video" />

        </androidx.swiperefreshlayout.widget.SwipeRefreshLayout>
```

The RecyclerView keeps its own id and attributes, so every existing
`binding.rcVideoList` reference and the `ConcatAdapter`/`GridLayoutManager` setup in
`initAdapters()` continues to work unchanged.

## 2. `app/src/main/java/com/arny/mobilecinema/presentation/home/HomeFragment.kt`

Wire the swipe gesture to a real data update in `initListeners()`:

```kotlin
    /** Sets up button listeners. */
    private fun initListeners() {
        binding.swipeRefresh.setOnRefreshListener {
            viewModel.downloadData(force = false)
        }
        binding.btnUpdateList.setOnClickListener {
            viewModel.loadMovies()
        }
        binding.homeContentTypeGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (syncingContentTypeSelection) return@addOnButtonCheckedListener
            if (!isChecked) return@addOnButtonCheckedListener
            when (checkedId) {
                R.id.btnHomeContentAll -> HomeContentType.ALL
                R.id.btnHomeContentCinema -> HomeContentType.CINEMA
                R.id.btnHomeContentSerial -> HomeContentType.SERIAL
                else -> null
            }?.let(viewModel::setContentType)
        }
    }
```

Then stop the spinner when paging settles. In the existing `addLoadStateListener` block
in `initAdapters()`, clear the refreshing flag on both terminal states:

```kotlin
                refresh is LoadState.Error -> {
                    binding.swipeRefresh.isRefreshing = false
                    binding.pbLoading.isVisible = false
                    binding.llEmptyState.isVisible = false
                    binding.errorView.isVisible = !hasAnyVisibleContent()
                    val errorMsg = refresh.error.localizedMessage
                        ?: getString(R.string.error_loading_data)
                    binding.tvErrorMessage.text = errorMsg
                }
                refresh is LoadState.NotLoading -> {
                    binding.swipeRefresh.isRefreshing = false
                    binding.pbLoading.isVisible = false
                    binding.errorView.isVisible = false
                    updateHighlightsVisibility()
                    val isEmpty = !hasAnyVisibleContent()
                    binding.llEmptyState.isVisible = isEmpty
                    binding.btnUpdateList.isVisible = isEmpty
                    binding.tvEmptyView.isVisible = isEmpty
                    hasQuery = !isEmpty
                    updateMenuVisibility()
                }
```

Since `pbLoading` and the swipe spinner would otherwise both appear during a
swipe-triggered load, suppress the centre spinner while the user is already seeing the
swipe indicator:

```kotlin
                refresh is LoadState.Loading -> {
                    binding.pbLoading.isVisible =
                        !hasAnyVisibleContent() && !binding.swipeRefresh.isRefreshing
                    binding.llEmptyState.isVisible = false
                    binding.errorView.isVisible = false
                }
```

Finally, cancel the indicator when the fragment goes away so a pending refresh does not
leak a spinning view back into a restored fragment — add to the existing `onPause()`:

```kotlin
    override fun onPause() {
        super.onPause()
        binding.swipeRefresh.isRefreshing = false
        ...
    }
```

## Why `downloadData` rather than `loadMovies`

`loadMovies()`/`reloadList()` both funnel into `refreshHome()` → `emitCurrentSearch()`,
which re-runs the paging query against Room. `MainPagingSource` reads exclusively from
the local database, so re-querying an unchanged database returns an identical list —
which is exactly the "nothing changes on refresh" symptom. `downloadData(force)` is the
only path that goes through `dataUpdateInteractor` to fetch new content, so that is what
a pull-to-refresh gesture has to call.

I passed `force = false` so the existing update-date check still applies and a swipe
doesn't force a full re-download every time; pass `true` if you want the gesture to
always bypass that check.
