"""
Author Grouping for Video Search Results

This module provides the AuthorGrouper class that aggregates video hits
by author (UP主) for display in the UI.

Key Features:
    - Groups hits by author mid
    - Tracks multiple aggregation metrics (count, views, scores)
    - Tracks first appearance order to match video list ordering
    - Supports multiple sort fields for different display needs

Author Sort Fields:
    - first_appear_order: Order by first video appearance (matches video list)
    - sum_rank_score: Sum of rank scores (most active/relevant authors)
    - top_rank_score: Max rank score (best single video)
    - sum_view: Total views (most popular authors)
    - sum_count: Total videos (most prolific authors)
"""

from typing import Literal

from tclogger import dict_get

from ranks.constants import (
    AUTHOR_SORT_FIELD_TYPE,
    AUTHOR_SORT_FIELD,
    AUTHOR_GROUP_LIMIT,
)


class AuthorGrouper:
    """Groups video hits by author for UI display.

    This class aggregates video search results by author (UP主) and provides
    various sort orders for the author list.

    IMPORTANT for UI Integration:
    - When sort_field="first_appear_order", the returned author dict preserves
      the order in which authors first appear in the video list.
    - This ensures the author list matches the video list order for
      "综合排序" (comprehensive sorting) mode.

    Example:
        >>> grouper = AuthorGrouper()
        >>> authors = grouper.group(
        ...     hits=search_results["hits"],
        ...     sort_field="first_appear_order",  # Match video order
        ...     limit=25,
        ... )
        >>> # authors is an OrderedDict with authors in first-appearance order
    """

    def group(
        self,
        hits: list[dict],
        sort_field: AUTHOR_SORT_FIELD_TYPE = AUTHOR_SORT_FIELD,
        limit: int = AUTHOR_GROUP_LIMIT,
    ) -> dict:
        """Group hits by author and sort by specified field.

        Args:
            hits: List of video hit documents.
            sort_field: Field to sort grouped authors by:
                - "first_appear_order": Order by when author first appears in hits.
                  This ensures author order matches the video list order.
                  RECOMMENDED for "综合排序" display mode.
                - "sum_rank_score": Sum of rank_scores (most active/relevant).
                - "top_rank_score": Max rank_score (best single video).
                - "sum_view": Total views (most popular).
                - "sum_count": Total videos (most prolific).
            limit: Maximum number of author groups to return.

        Returns:
            Dict of author groups keyed by mid, in sorted order.
            Each author group contains:
                - mid: Author ID
                - name: Author name (from most recent video)
                - latest_pubdate: Most recent video pubdate
                - sum_view: Total views across all videos
                - sum_sort_score: Sum of sort scores
                - sum_rank_score: Sum of rank scores
                - top_rank_score: Max rank score
                - first_appear_order: Index of first video in hits
                - sum_count: Number of videos
                - hits: List of video hits by this author
                - face: Author avatar URL (if available)
        """
        group_res = {}
        first_appear_idx = {}  # Track first appearance index for each author

        for idx, hit in enumerate(hits):
            name = dict_get(hit, "owner.name", None)
            mid = dict_get(hit, "owner.mid", None)
            pubdate = dict_get(hit, "pubdate") or 0
            view = dict_get(hit, "stat.view") or 0
            sort_score = dict_get(hit, "sort_score") or 0
            rank_score = dict_get(hit, "rank_score") or 0

            if mid is None or name is None:
                continue

            # Track first appearance index for this author
            if mid not in first_appear_idx:
                first_appear_idx[mid] = idx

            item = group_res.get(mid, None)
            if item is None:
                group_res[mid] = {
                    "mid": mid,
                    "name": name,
                    "latest_pubdate": pubdate,
                    "sum_view": view,
                    "sum_sort_score": sort_score,
                    "sum_rank_score": rank_score,
                    "top_rank_score": rank_score,
                    "first_appear_order": idx,  # Store first appearance index
                    "sum_count": 1,
                    "hits": [hit],
                }
            else:
                # Update with latest video info if more recent
                latest_pubdate = group_res[mid]["latest_pubdate"]
                if pubdate > latest_pubdate:
                    group_res[mid]["latest_pubdate"] = pubdate
                    group_res[mid]["name"] = name

                # Aggregate metrics
                group_res[mid]["sum_view"] = (group_res[mid]["sum_view"] or 0) + view
                group_res[mid]["sum_sort_score"] = (
                    group_res[mid]["sum_sort_score"] or 0
                ) + sort_score
                group_res[mid]["sum_rank_score"] = (
                    group_res[mid]["sum_rank_score"] or 0
                ) + rank_score
                group_res[mid]["top_rank_score"] = max(
                    group_res[mid]["top_rank_score"] or 0, rank_score
                )
                group_res[mid]["hits"].append(hit)
                group_res[mid]["sum_count"] += 1

        # Sort by specified field
        # For first_appear_order: lower index = earlier appearance = higher priority (ascending)
        # For other fields: higher value = higher priority (descending)
        if sort_field == "first_appear_order":
            sorted_items = sorted(
                group_res.items(),
                key=lambda item: item[1].get(sort_field, float("inf")),
                reverse=False,  # Ascending: lower index first
            )[:limit]
        else:
            sorted_items = sorted(
                group_res.items(),
                key=lambda item: item[1].get(sort_field, 0),
                reverse=True,  # Descending: higher value first
            )[:limit]

        # Convert back to dict (preserves order in Python 3.7+)
        group_res = dict(sorted_items)

        return group_res

    def group_from_search_result(
        self,
        search_res: dict,
        sort_field: AUTHOR_SORT_FIELD_TYPE = AUTHOR_SORT_FIELD,
        limit: int = AUTHOR_GROUP_LIMIT,
    ) -> dict:
        """Group authors from a search result dict.

        Convenience method that extracts hits from search_res.

        Args:
            search_res: Search result dict containing "hits" list.
            sort_field: Field to sort grouped authors by.
            limit: Maximum number of author groups to return.

        Returns:
            Dict of author groups keyed by mid.
        """
        hits = search_res.get("hits", [])
        return self.group(hits, sort_field=sort_field, limit=limit)

    def add_user_faces(self, group_res: dict, user_docs: dict) -> dict:
        """Add user face (avatar) URLs to author groups.

        Args:
            group_res: Author groups dict from group().
            user_docs: Dict mapping mid to user doc with "face" field.

        Returns:
            Updated group_res with "face" field added to each author.
        """
        for mid, author_info in group_res.items():
            user_doc = user_docs.get(mid, {})
            author_info["face"] = user_doc.get("face", "")
        return group_res

    def group_as_list(
        self,
        hits: list[dict],
        sort_field: AUTHOR_SORT_FIELD_TYPE = AUTHOR_SORT_FIELD,
        limit: int = AUTHOR_GROUP_LIMIT,
    ) -> list[dict]:
        """Group hits by author and return as sorted list.

        This method returns a LIST instead of a dict to ensure order preservation
        across JSON serialization/deserialization (network transport).

        IMPORTANT: Use this method for API responses to frontend!
        Dict key ordering is NOT guaranteed after JSON transport,
        but list ordering IS guaranteed.

        Args:
            hits: List of video hit documents.
            sort_field: Field to sort grouped authors by.
            limit: Maximum number of author groups to return.

        Returns:
            List of author groups, sorted by sort_field.
            Order is guaranteed to be preserved in JSON.
        """
        group_dict = self.group(hits, sort_field=sort_field, limit=limit)
        # Convert dict values to list - order is preserved from sorted dict
        return list(group_dict.values())

    def group_from_search_result_as_list(
        self,
        search_res: dict,
        sort_field: AUTHOR_SORT_FIELD_TYPE = AUTHOR_SORT_FIELD,
        limit: int = AUTHOR_GROUP_LIMIT,
    ) -> list[dict]:
        """Group authors from search result and return as sorted list.

        IMPORTANT: Use this method for API responses to frontend!

        Args:
            search_res: Search result dict containing "hits" list.
            sort_field: Field to sort grouped authors by.
            limit: Maximum number of author groups to return.

        Returns:
            List of author groups, sorted by sort_field.
        """
        hits = search_res.get("hits", [])
        return self.group_as_list(hits, sort_field=sort_field, limit=limit)

    def add_user_faces_to_list(
        self, authors_list: list[dict], user_docs: dict
    ) -> list[dict]:
        """Add user face (avatar) URLs to author list.

        Args:
            authors_list: List of author groups from group_as_list().
            user_docs: Dict mapping mid to user doc with "face" field.

        Returns:
            Updated authors_list with "face" field added to each author.
        """
        for author_info in authors_list:
            mid = author_info.get("mid")
            user_doc = user_docs.get(mid, {})
            author_info["face"] = user_doc.get("face", "")
        return authors_list
