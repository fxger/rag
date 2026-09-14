"""
Update project state document.
Run: python code_state/update_state.py
"""

import os
from datetime import datetime

def count_files(directory, ext=None):
    """Count files in directory."""
    count = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if ext is None or f.endswith(ext):
                count += 1
    return count

def get_milvus_stats():
    """Get Milvus collection stats."""
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://192.168.86.132:19530")
        client.load_collection("enterprise_rag")
        stats = client.get_collection_stats("enterprise_rag")
        return stats.get("row_count", 0)
    except:
        return -1

def main():
    project_root = "D:\\Desktop\\RAG"
    
    print("=" * 60)
    print("  Enterprise RAG System - State Update")
    print("=" * 60)
    print()
    
    # Count files
    py_files = count_files(os.path.join(project_root, "src"), ".py")
    test_files = count_files(os.path.join(project_root, "tests"), ".py")
    doc_files = count_files(os.path.join(project_root, "docs"), ".md")
    data_files = count_files(os.path.join(project_root, "test_data"))
    
    # Get Milvus stats
    milvus_count = get_milvus_stats()
    
    print("Source files (src/):    %d" % py_files)
    print("Test files (tests/):    %d" % test_files)
    print("Doc files (docs/):      %d" % doc_files)
    print("Data files (test_data/):%d" % data_files)
    print("Milvus vectors:         %d" % milvus_count)
    print()
    
    # Update timestamp in PROJECT_STATE.md
    state_file = os.path.join(project_root, "code_state", "PROJECT_STATE.md")
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Update timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = content.replace(
            "> 生成时间: 2026-09-10 21:58",
            "> 生成时间: %s" % now
        )
        
        with open(state_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        print("Updated timestamp: %s" % now)
    
    print()
    print("State document updated successfully!")

if __name__ == "__main__":
    main()
