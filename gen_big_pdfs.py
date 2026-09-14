import os, random, textwrap
random.seed(42)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

D = r"D:\Desktop\RAG\test_data"

def gen_pdf(fn, title, paragraphs):
    fp = os.path.join(D, fn)
    c = canvas.Canvas(fp, pagesize=A4)
    w, h = A4
    y = h - 2*cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, title)
    y -= 1*cm
    c.setFont("Helvetica", 9)
    for p in paragraphs:
        if y < 3*cm:
            c.showPage()
            y = h - 2*cm
        for line in textwrap.wrap(p, 100):
            if y < 2*cm:
                c.showPage()
                y = h - 2*cm
            c.drawString(2*cm, y, line)
            y -= 0.4*cm
        y -= 0.3*cm
    c.save()
    sz = os.path.getsize(fp)
    print(f"  {fn}: {sz} bytes ({sz/1024:.1f} KB)")

# ========== PDF 1: Market Analysis ==========
p1 = []
p1.append("2025 Annual Market Analysis Report - XX Technology Co., Ltd.")
p1.append("Published: September 2025 | Classification: Confidential")
p1.append("")
for i in range(50):
    p1.append(f"Section paragraph {i+1}: The enterprise AI market continues to grow rapidly in 2025, with total spending reaching 150 billion USD globally, representing a 35% increase year-over-year. This growth is driven by digital transformation initiatives across all industry verticals, increasing adoption of cloud-based AI services, and the maturation of foundation models for enterprise applications. Our company, XX Technology, has successfully captured 2.3% of this growing market, positioning us among the top 20 enterprise AI vendors globally.")
for i in range(50):
    p1.append(f"Detail paragraph {i+1}: Revenue breakdown by geography shows North America contributing 40% of global AI spending, followed by Asia-Pacific at 30%, Europe at 20%, and the rest of the world at 10%. Within North America, the United States accounts for 85% of spending, with Canada at 10% and Mexico at 5%. The enterprise segment represents 65% of total spending, with small and medium businesses accounting for the remaining 35%. Key growth verticals include Financial Services (28%), Healthcare (18%), Manufacturing (15%), Retail (12%), and Technology (10%).")
for i in range(50):
    p1.append(f"Analysis paragraph {i+1}: Our competitive analysis reveals that Tier 1 vendors (OpenAI, Google, Microsoft, Amazon) control 45% of the market through their cloud platforms and foundation models. Tier 2 vendors (Anthropic, Cohere, Mistral, IBM) capture 20% with specialized enterprise solutions. Our tier (Tier 3) with 50+ competitors accounts for 25% of the market, with the remaining 10% distributed across open-source and custom-built solutions. Our key differentiators include vertical-specific solutions, on-premise deployment capabilities, and superior customer support with 4.6/5.0 satisfaction scores.")
gen_pdf("market_analysis_2025.pdf", "2025 Market Analysis Report", p1)

# ========== PDF 2: Tech Whitepaper ==========
p2 = []
p2.append("Enterprise RAG System: Architecture and Best Practices")
p2.append("Technical White Paper - Version 2.0 | Published: September 2025")
p2.append("")
for i in range(50):
    p2.append(f"Architecture paragraph {i+1}: The enterprise RAG architecture consists of five core layers designed for scalability, reliability, and security. Layer 1 handles Document Ingestion with support for 1000+ file formats through Apache Tika integration, including PDF, DOCX, XLSX, HTML, and custom enterprise formats. The pipeline processes 50 million documents monthly with 99.8% extraction accuracy and average processing time of 2.3 seconds per page. Quality assessment includes readability scoring, deduplication using MinHash algorithms, language detection across 50+ languages, and PII scanning for compliance.")
for i in range(50):
    p2.append(f"Technology paragraph {i+1}: Layer 2 manages Text Processing and Chunking with adaptive strategies including fixed-size chunking (512 tokens with 100 token overlap), semantic chunking using sentence embeddings for boundary detection, recursive character splitting respecting document structure, and document-aware chunking that preserves sections, paragraphs, and tables. Layer 3 provides Vector Storage and Indexing with support for multiple database backends: Milvus (distributed, hybrid search), Pinecone (managed SaaS), Weaviate (GraphQL API), and Qdrant (high-performance Rust implementation). Our benchmark testing shows Milvus achieving 12,000 QPS at 1 million vectors with 1024 dimensions.")
for i in range(50):
    p2.append(f"Performance paragraph {i+1}: Production deployment across 500+ enterprise customers demonstrates the following metrics: Query latency P50 120ms, P95 350ms, P99 800ms. Throughput sustained 10,000 QPS, burst 50,000 QPS. Retrieval quality Recall@10 92%, Precision@5 85%, MRR 0.78. System availability 99.97% monthly average. Document processing capacity 100,000 documents per hour. Cost per query $0.0002 including compute and embedding API costs. Storage efficiency 1.2x raw text size with embeddings. Total cost of ownership 40% lower than SaaS alternatives with break-even at 18 months for deployments exceeding 10 million documents.")
gen_pdf("enterprise_rag_whitepaper.pdf", "Enterprise RAG Whitepaper", p2)

# ========== PDF 3: Product Roadmap ==========
p3 = []
p3.append("Product Roadmap - RAG Platform V3.0")
p3.append("Confidential - Internal Use Only | Last Updated: September 2025")
p3.append("")
for i in range(50):
    p3.append(f"Roadmap paragraph {i+1}: Q4 2025 Foundation Phase focuses on three critical features. First, Multimodal Document Processing with GPT-4V integration for image understanding, table extraction with structure preservation, and chart/diagram comprehension. Target: 95% accuracy on image-to-text conversion with table structure fully preserved. Second, Advanced Retrieval Engine with hybrid search combining vector similarity (0.7 weight) and BM25 keyword matching (0.3 weight), learned reranking using cross-encoder models, and multi-step retrieval for complex queries requiring information from multiple documents. Third, Enterprise Security Hardening including SSO integration via SAML 2.0 and OAuth 2.0, document-level access control with role-based permissions, and comprehensive audit trail for all operations to achieve SOC 2 Type II compliance.")
for i in range(50):
    p3.append(f"Feature paragraph {i+1}: Q1 2026 Intelligence Phase introduces Knowledge Graph Integration with automatic entity extraction and linking achieving 90% accuracy, graph-based reasoning for complex relational queries with sub-second traversal, and visual knowledge graph explorer for interactive data exploration. Conversational RAG enables multi-turn dialogue with context retention for 10+ turns, clarification questions for ambiguous queries, and proactive information gathering to resolve complex support scenarios. Analytics Dashboard provides real-time query analytics and trending topics, document usage and relevance metrics, user behavior insights, and ROI calculator for business stakeholder reporting.")
for i in range(50):
    p3.append(f"Strategy paragraph {i+1}: Q2 2026 Scale Phase delivers Multi-Region Deployment with active-active architecture across regions, data residency compliance for GDPR and PIPL regulations, and cross-region replication with conflict resolution achieving <100ms latency and zero data loss. Plugin Architecture enables custom parser plugins, retrieval strategy plugins, response post-processing plugins, and marketplace for community-contributed plugins with security sandboxing. Mobile SDK provides iOS and Android native implementations with offline mode supporting 1000 documents and voice input/output at 95% accuracy. Q3 2026 Enterprise Phase adds AI Agent Framework with tool use, multi-agent orchestration, human-in-the-loop workflows, Compliance Center with regulatory templates and automated policy checking, and Global Partner Portal for onboarding, certification, and revenue sharing.")
gen_pdf("product_roadmap_v3.pdf", "Product Roadmap V3.0", p3)

total = sum(os.path.getsize(os.path.join(D,f)) for f in os.listdir(D) if os.path.isfile(os.path.join(D,f)))
print(f"\nTotal dir size: {total/1024:.0f} KB")
