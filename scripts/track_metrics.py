#!/usr/bin/env python3
import os
import json
import urllib.request
import urllib.error
from datetime import datetime

REPO_OWNER = "CoderDayton"
REPO_NAME = "tinyvecdb"
GITHUB_API_URL = "https://api.github.com"

def get_github_metrics(token=None):
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}"
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        return {
            "stars": data.get("stargazers_count", 0),
            "forks": data.get("forks_count", 0),
            "watchers": data.get("subscribers_count", 0),
            "open_issues": data.get("open_issues_count", 0),
        }
    except urllib.error.HTTPError as e:
        print(f"Error fetching repo metrics: {e}")
        return None

def get_sponsor_count(token=None):
    # Note: Accurate sponsor count requires GraphQL API and a token with read:user/read:org scope
    if not token:
        return "N/A (Requires GITHUB_TOKEN)"
    
    query = """
    {
      user(login: "%s") {
        sponsorshipsAsMaintainer {
          totalCount
        }
      }
    }
    """ % REPO_OWNER
    
    url = "https://api.github.com/graphql"
    headers = {
        "Authorization": f"bearer {token}",
        "Content-Type": "application/json"
    }
    data = json.dumps({"query": query}).encode("utf-8")
    
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result.get("data", {}).get("user", {}).get("sponsorshipsAsMaintainer", {}).get("totalCount", 0)
    except Exception as e:
        print(f"Error fetching sponsor count: {e}")
        return "Error"

def main():
    token = os.environ.get("GITHUB_TOKEN")
    
    print(f"--- Metrics for {REPO_OWNER}/{REPO_NAME} ---")
    print(f"Date: {datetime.now().isoformat()}")
    
    metrics = get_github_metrics(token)
    if metrics:
        print(f"⭐ Stars:    {metrics['stars']}")
        print(f"🍴 Forks:    {metrics['forks']}")
        print(f"👀 Watchers: {metrics['watchers']}")
        print(f"🐞 Issues:   {metrics['open_issues']}")
    
    sponsors = get_sponsor_count(token)
    print(f"💖 Sponsors: {sponsors}")

if __name__ == "__main__":
    main()
