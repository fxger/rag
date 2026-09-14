import httpx

# Search GitHub for RAG test datasets
url = "https://api.github.com/search/repositories"
params = {"q": "RAG test dataset documents", "sort": "stars", "per_page": 10}
r = httpx.get(url, params=params, timeout=10)
data = r.json()

print("Found repositories:")
for item in data.get("items", [])[:10]:
    name = item["full_name"]
    stars = item["stargazers_count"]
    desc = (item.get("description") or "No description")[:80]
    print("  %s (%d stars)" % (name, stars))
    print("    %s" % desc)
    print("    %s" % item["html_url"])
    print()
