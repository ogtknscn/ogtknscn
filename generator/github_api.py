"""GitHub API client for fetching user stats and language data."""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)


class GitHubAPI:
    """Fetches GitHub stats via GraphQL (with token) or REST (fallback)."""

    GRAPHQL_URL = "https://api.github.com/graphql"
    REST_URL = "https://api.github.com"

    def __init__(self, username: str, token: str = None):
        self.username = username
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an HTTP request with rate-limit awareness and retry.

        Checks X-RateLimit-Remaining after each response.
        On 403 rate-limit, waits until reset and retries once.
        """
        kwargs.setdefault("headers", self.headers)
        kwargs.setdefault("timeout", 15)

        resp = requests.request(method, url, **kwargs)

        # Check rate limit headers
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining is not None and int(remaining) < 10:
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
            logger.warning(
                "GitHub API rate limit low: %s remaining (resets at %s)",
                remaining,
                time.strftime("%H:%M:%S", time.localtime(reset_ts)),
            )

        # Retry once on rate-limit 403
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_ts - int(time.time()), 1)
            logger.warning("Rate limited. Waiting %ds for reset...", wait)
            time.sleep(wait)
            resp = requests.request(method, url, **kwargs)

        return resp

    def fetch_stats(self) -> dict:
        """Fetch user statistics. Uses GraphQL if token available, REST otherwise."""
        if self.token:
            return self._fetch_stats_graphql()
        return self._fetch_stats_rest()

    def _fetch_stats_graphql(self) -> dict:
        """Fetch stats via GraphQL for accurate counts including private contributions."""
        query = """
        query($username: String!) {
          user(login: $username) {
            createdAt
            pullRequests {
              totalCount
            }
            issues {
              totalCount
            }
            repositories(ownerAffiliations: OWNER, first: 100) {
              totalCount
              nodes {
                stargazerCount
              }
            }
          }
        }
        """
        try:
            resp = self._request(
                "POST",
                self.GRAPHQL_URL,
                json={"query": query, "variables": {"username": self.username}},
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            logger.warning("GraphQL request timed out, falling back to REST.")
            return self._fetch_stats_rest()
        except requests.exceptions.HTTPError as e:
            logger.warning("GraphQL HTTP error (%s), falling back to REST.", e)
            return self._fetch_stats_rest()

        data = resp.json()

        if "errors" in data:
            logger.warning("GraphQL errors: %s", data["errors"])
            return self._fetch_stats_rest()

        user = data["data"]["user"]
        repos = user["repositories"]

        total_stars = sum(n["stargazerCount"] for n in repos["nodes"])

        try:
            total_commits = self._fetch_total_commits(user["createdAt"])
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            logger.warning("Could not fetch lifetime commit total (%s), falling back to REST.", e)
            return self._fetch_stats_rest()

        return {
            "commits": total_commits,
            "stars": total_stars,
            "prs": user["pullRequests"]["totalCount"],
            "issues": user["issues"]["totalCount"],
            "repos": repos["totalCount"],
        }

    def _fetch_total_commits(self, created_at: str) -> int:
        """Sum commit contributions across every year since account creation.

        A single contributionsCollection query caps its from/to window at one
        year and defaults to the trailing 365 days when no window is given,
        so it can't report a lifetime total on its own - this issues one
        aliased field per year and adds them up.
        """
        start = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)

        ranges = []
        cursor = start
        while cursor < now:
            window_end = min(cursor + timedelta(days=365), now)
            ranges.append((cursor.isoformat(), window_end.isoformat()))
            cursor = window_end

        if not ranges:
            return 0

        fragment = "\n".join(
            f'y{i}: contributionsCollection(from: "{f}", to: "{t}") {{ '
            f"totalCommitContributions restrictedContributionsCount }}"
            for i, (f, t) in enumerate(ranges)
        )
        query = f"""
        query($username: String!) {{
          user(login: $username) {{
            {fragment}
          }}
        }}
        """
        resp = self._request(
            "POST",
            self.GRAPHQL_URL,
            json={"query": query, "variables": {"username": self.username}},
        )
        resp.raise_for_status()

        data = resp.json()
        if "errors" in data:
            raise ValueError(data["errors"])

        user = data["data"]["user"]
        return sum(
            user[f"y{i}"]["totalCommitContributions"] + user[f"y{i}"]["restrictedContributionsCount"]
            for i in range(len(ranges))
        )

    def _fetch_stats_rest(self) -> dict:
        """Fallback: fetch stats via REST API (public data only)."""
        user_resp = self._request(
            "GET", f"{self.REST_URL}/users/{self.username}"
        )
        user_resp.raise_for_status()
        user_data = user_resp.json()

        # Fetch repos to count stars
        total_stars = 0
        for repos in self._paginate_repos():
            total_stars += sum(r.get("stargazers_count", 0) for r in repos)

        # Estimate commits from events (rough approximation without token)
        events_resp = self._request(
            "GET",
            f"{self.REST_URL}/users/{self.username}/events/public",
            params={"per_page": 100},
        )
        events_resp.raise_for_status()
        events = events_resp.json()
        commit_count = sum(
            len(e.get("payload", {}).get("commits", []))
            for e in events
            if e.get("type") == "PushEvent"
        )

        # Fetch actual PR count via Search API
        pr_count = self._search_count(f"author:{self.username} type:pr")

        # Fetch actual issue count via Search API
        issue_count = self._search_count(f"author:{self.username} type:issue")

        return {
            "commits": commit_count,
            "stars": total_stars,
            "prs": pr_count,
            "issues": issue_count,
            "repos": user_data.get("public_repos", 0),
        }

    def _paginate_repos(self):
        """Yield pages of owned repos from the REST API."""
        page = 1
        while True:
            repos_resp = self._request(
                "GET",
                f"{self.REST_URL}/users/{self.username}/repos",
                params={"per_page": 100, "page": page, "type": "owner"},
            )
            repos_resp.raise_for_status()
            repos = repos_resp.json()
            if not repos:
                break
            yield repos
            if len(repos) < 100:
                break
            page += 1

    def _search_count(self, query: str) -> int:
        """Use the GitHub Search API to get a total_count for a query."""
        try:
            resp = self._request(
                "GET",
                f"{self.REST_URL}/search/issues",
                params={"q": query, "per_page": 1},
            )
            if resp.status_code == 200:
                return resp.json().get("total_count", 0)
            logger.warning("Search API returned %d for query '%s'", resp.status_code, query)
        except requests.exceptions.RequestException as e:
            logger.warning("Search API failed for '%s': %s", query, e)
        return 0

    def fetch_languages(self) -> dict:
        """Fetch language byte counts aggregated across all owned non-fork repos."""
        languages = {}
        for repos in self._paginate_repos():
            for repo in repos:
                if repo.get("fork"):
                    continue
                try:
                    lang_resp = self._request("GET", repo["languages_url"])
                    if lang_resp.status_code == 200:
                        for lang, bytes_count in lang_resp.json().items():
                            languages[lang] = languages.get(lang, 0) + bytes_count
                    else:
                        logger.warning(
                            "Could not fetch languages for %s (HTTP %d)",
                            repo.get("full_name", "unknown"),
                            lang_resp.status_code,
                        )
                except requests.exceptions.RequestException as e:
                    logger.warning(
                        "Error fetching languages for %s: %s",
                        repo.get("full_name", "unknown"),
                        e,
                    )
        return languages
