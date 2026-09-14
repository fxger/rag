from pymilvus import MilvusClient
client = MilvusClient(uri="http://192.168.86.132:19530")

# List collections
collections = client.list_collections()
print("Collections:", collections)

# Check enterprise_rag collection
if "enterprise_rag" in collections:
    info = client.describe_collection("enterprise_rag")
    print("\nenterprise_rag schema:")
    for field in info["fields"]:
        fname = field["name"]
        ftype = field["type"]
        fparams = field.get("params", {})
        print("  %s: type=%s, params=%s" % (fname, ftype, fparams))
else:
    print("\nenterprise_rag collection not found, creating...")
    # Will be created on first use
