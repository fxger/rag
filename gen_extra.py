"""Generate extra PDF and Excel test files"""
import os, random
random.seed(42)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

D = r"D:\Desktop\RAG\test_data"

def wrap_text(c, lines, x, y, h, font="Helvetica", size=9, max_w=95, gap=0.45):
    c.setFont(font, size)
    for line in lines:
        if y < 2*cm:
            c.showPage(); y = h - 2*cm
        words = line.split(); cur = ""
        for w in words:
            t = (cur + " " + w) if cur else w
            if len(t) > max_w:
                c.drawString(x, y, cur); y -= gap*cm; cur = w
            else:
                cur = t
        if cur:
            c.drawString(x, y, cur); y -= gap*cm
    return y

# ===== PDF 1: Market Analysis =====
def pdf_market():
    fp = os.path.join(D, "market_analysis_2025.pdf")
    c = canvas.Canvas(fp, pagesize=A4); w, h = A4
    secs = [
        ("1. Executive Summary", [
            "The enterprise AI market reached 150 billion USD in 2025, +35% YoY.",
            "Key drivers: digital transformation, cost reduction, competitive advantage.",
            "Our market share: 2.3%, top 20 enterprise AI vendor.",
            "Customer satisfaction: 4.6/5.0, NPS: 52.",
        ]),
        ("2. Market Overview", [
            "TAM: 500B by 2030. SAM: 150B in 2025. SOM: 15B.",
            "North America 40%, APAC 30%, Europe 20%, Others 10%.",
            "Verticals: FinServ 28%, Healthcare 18%, Mfg 15%, Retail 12%.",
        ]),
        ("3. Competitive Landscape", [
            "Tier1: OpenAI, Google, MS, Amazon - cloud AI.",
            "Tier2: Anthropic, Cohere, Mistral - enterprise solutions.",
            "Tier3: Startups in niches and regions.",
            "Our edge: vertical solutions, on-prem, superior support.",
        ]),
        ("4. Product Performance", [
            "RAG V2.0: 1200 deployments, 99.2% uptime, 180ms latency.",
            "Doc Pipeline: 50M docs/mo, 99.8% accuracy.",
            "Knowledge Graph: 10B triples, sub-second traversal.",
            "API: 50K QPS, 99.99% availability.",
        ]),
        ("5. Regional Analysis", [
            "NA: 25M rev, 35% growth, 180 clients.",
            "Europe: 15M rev, 28% growth, 120 clients.",
            "APAC: 18M rev, 52% growth, 150 clients.",
            "LATAM: 5M rev, 40% growth.",
        ]),
        ("6. Financial Metrics", [
            "Revenue: 52M (104% of target). Gross margin: 68%.",
            "CAC: 45K, LTV: 580K, LTV/CAC: 12.9x.",
            "MRR: 4.3M, retention 98%. ARR: 52M, NRR 135%.",
        ]),
        ("7. Risk Assessment", [
            "High: EU AI Act, China AI regulations.",
            "Medium: Talent shortage, 30% salary inflation.",
            "Medium: Cloud dependency (60% AWS/Azure).",
            "Low: Quantum computing disruption.",
        ]),
        ("8. Strategic Recommendations", [
            "1. Invest 10M in R&D for multimodal capabilities.",
            "2. Expand APAC team by 50 FTEs.",
            "3. Launch partner program (100 SIs).",
            "4. Achieve SOC2 Type II, ISO 27001.",
            "5. AI governance framework for regulated industries.",
        ]),
    ]
    y = h - 2*cm
    for title, lines in secs:
        if y < 4*cm: c.showPage(); y = h - 2*cm
        c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, title); y -= 0.8*cm
        y = wrap_text(c, lines, 2.5*cm, y, h)
        y -= 0.3*cm
    c.save()
    print("  market_analysis_2025.pdf:", os.path.getsize(fp), "bytes")

# ===== PDF 2: Tech Whitepaper =====
def pdf_whitepaper():
    fp = os.path.join(D, "enterprise_rag_whitepaper.pdf")
    c = canvas.Canvas(fp, pagesize=A4); w, h = A4
    secs = [
        ("1. Introduction", [
            "RAG combines LLMs with external knowledge for accurate, grounded responses.",
            "Benefits: reduced hallucination, real-time updates, source attribution, cost-effective.",
            "Challenges: scale (millions of docs), multi-format, hybrid search, security, HA.",
        ]),
        ("2. System Architecture", [
            "Layer 1 - Ingestion: Tika parsers, OCR, table extraction, metadata.",
            "Layer 2 - Processing: Adaptive chunking, semantic boundaries, overlap.",
            "Layer 3 - Storage: Milvus/Pinecone/Weaviate, IVF_FLAT/HNSW indexes.",
            "Layer 4 - Retrieval: Multi-stage, hybrid search, cross-encoder reranking.",
            "Layer 5 - Generation: Prompt engineering, streaming, citations, hallucination detection.",
        ]),
        ("3. Document Pipeline", [
            "Acquisition: Filesystem monitoring, API ingestion, web crawling.",
            "Parsing: MIME detection, Tika, custom parsers, fallbacks.",
            "Extraction: Text, tables (camelot), images, metadata.",
            "Quality: Scoring, dedup (MinHash), language detection, PII scanning.",
            "Chunking: Structure-aware, overlapping windows, entity extraction.",
        ]),
        ("4. Vector Database Guide", [
            "Milvus: Distributed, multiple indexes, hybrid search, open source.",
            "Pinecone: Fully managed SaaS, low ops overhead.",
            "Weaviate: GraphQL API, built-in vectorization, multi-modal.",
            "Qdrant: Rust implementation, rich filtering, efficient memory.",
            "Benchmark (1M vectors, 1024d): Milvus 12K QPS, Pinecone 8K, Qdrant 10K.",
        ]),
        ("5. Embedding Strategies", [
            "text-embedding-3: 3072d, high quality, commercial.",
            "bge-large-en: 1024d, fast, open source.",
            "qwen-embedding: 1024d, Chinese optimized.",
            "Best practices: domain fine-tuning, normalization, caching.",
        ]),
        ("6. Retrieval Optimization", [
            "Query processing: expansion, rewriting, HyDE, multi-query.",
            "Hybrid search: vector 0.6-0.8, BM25 0.2-0.4, RRF fusion.",
            "Re-ranking: Cross-encoder, ColBERT, LLM-based.",
            "Performance: ANN, sharding, caching, background optimization.",
        ]),
        ("7. Production Deployment", [
            "HA: Multi-replica, auto-failover, replication factor 3.",
            "Scale: Horizontal scaling, read replicas, auto-scaling.",
            "Security: RBAC, AES-256, TLS 1.3, audit logging, PII redaction.",
            "Monitoring: Prometheus, Jaeger, custom dashboards, alerting.",
            "DR: WAL, cross-region replication, RPO<1h, RTO<15min.",
        ]),
        ("8. Benchmarks", [
            "Latency: P50 120ms, P95 350ms, P99 800ms.",
            "Throughput: 10K QPS sustained, 50K burst.",
            "Quality: Recall@10 92%, Precision@5 85%, MRR 0.78.",
            "Uptime: 99.97%. Doc processing: 100K docs/hour.",
            "Cost: $0.0002/query, $0.10/GB/month storage.",
        ]),
    ]
    y = h - 2*cm
    for title, lines in secs:
        if y < 4*cm: c.showPage(); y = h - 2*cm
        c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, title); y -= 0.7*cm
        y = wrap_text(c, lines, 2.5*cm, y, h)
        y -= 0.3*cm
    c.save()
    print("  enterprise_rag_whitepaper.pdf:", os.path.getsize(fp), "bytes")

# ===== PDF 3: Product Roadmap =====
def pdf_roadmap():
    fp = os.path.join(D, "product_roadmap_v3.pdf")
    c = canvas.Canvas(fp, pagesize=A4); w, h = A4
    secs = [
        ("Q4 2025 - Foundation", [
            "Multimodal doc processing (GPT-4V), advanced retrieval, enterprise security.",
            "Timeline: Oct-Dec 2025. Teams: Doc Processing, Search, Security.",
        ]),
        ("Q1 2026 - Intelligence", [
            "Knowledge graph integration, conversational RAG, analytics dashboard.",
            "Timeline: Jan-Mar 2026. Teams: AI Research, NLP, Analytics.",
        ]),
        ("Q2 2026 - Scale", [
            "Multi-region deployment, plugin architecture, mobile SDK.",
            "Timeline: Apr-Jun 2026. Teams: Infrastructure, Platform, Mobile.",
        ]),
        ("Q3 2026 - Enterprise", [
            "AI agent framework, compliance center, partner portal.",
            "Timeline: Jul-Sep 2026. Teams: AI Research, Compliance, Partners.",
        ]),
        ("Resource Requirements", [
            "Engineering: 45 FTEs (current 32). Infra budget: 2.5M.",
            "Cloud: 1.2M, Vector DB: 500K, Security: 400K, Tools: 200K.",
        ]),
        ("Success Metrics", [
            "500+ enterprise customers, 99.95% uptime, P95 < 500ms.",
            "Revenue: 75M ARR, 500 new accounts, NPS > 55.",
        ]),
    ]
    y = h - 2*cm
    for title, lines in secs:
        if y < 4*cm: c.showPage(); y = h - 2*cm
        c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, title); y -= 0.7*cm
        y = wrap_text(c, lines, 2.5*cm, y, h)
        y -= 0.3*cm
    c.save()
    print("  product_roadmap_v3.pdf:", os.path.getsize(fp), "bytes")

# ===== Excel 1: Customer Sales =====
def xlsx_customer():
    fp = os.path.join(D, "customer_sales_data.xlsx")
    wb = Workbook()
    hfont = Font(bold=True, color="FFFFFF"); hfill = PatternFill("solid", fgColor="4472C4")
    bdr = Border(*(Side(style="thin"),)*4)
    ind = ["Financial","Healthcare","Manufacturing","Retail","Technology","Government"]
    reg = ["NA","Europe","APAC","LATAM","ME"]
    sizes = ["Enterprise","Mid-Market","SMB"]
    statuses = ["Active","Renewing","Churning","Onboarding"]
    # Sheet 1: Customer List
    ws = wb.active; ws.title = "Customers"
    hdrs = ["ID","Company","Industry","Region","Size","Contract","Start","End","Status","NPS"]
    for i,h in enumerate(hdrs,1):
        c = ws.cell(1,i,h); c.font=hfont; c.fill=hfill; c.border=bdr
    for r in range(2, 152):
        ws.cell(r,1,f"CUST-{1000+r-2:04d}")
        ws.cell(r,2,f"Company_{r-1}")
        ws.cell(r,3,random.choice(ind)); ws.cell(r,4,random.choice(reg))
        ws.cell(r,5,random.choice(sizes))
        ws.cell(r,6,round(random.uniform(50000,5000000),2))
        ws.cell(r,7,f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
        ws.cell(r,8,f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
        ws.cell(r,9,random.choice(statuses)); ws.cell(r,10,random.randint(1,10))
        for col in range(1,11): ws.cell(r,col).border=bdr
    for i in range(1,11): ws.column_dimensions[chr(64+i)].width=16
    # Sheet 2: Monthly Revenue
    ws2 = wb.create_sheet("Revenue")
    hdrs2 = ["Month","New","Renewal","Expansion","Churn","Net","Cumulative","YoY"]
    for i,h in enumerate(hdrs2,1):
        c = ws2.cell(1,i,h); c.font=hfont; c.fill=hfill; c.border=bdr
    months = [450000,380000,120000,-45000, 520000,410000,95000,-38000,
              680000,450000,180000,-52000, 420000,520000,110000,-41000,
              550000,480000,145000,-55000, 720000,510000,210000,-48000,
              480000,550000,165000,-62000, 610000,580000,190000,-51000,
              580000,620000,225000,-44000, 650000,590000,185000,-57000,
              720000,640000,240000,-63000, 850000,710000,320000,-71000]
    cum = 0
    for m in range(12):
        n,r,e,ch = months[m*4:m*4+4]
        net = n+r+e+ch; cum += net
        ws2.cell(m+2,1,f"2025-{m+1:02d}")
        ws2.cell(m+2,2,n); ws2.cell(m+2,3,r); ws2.cell(m+2,4,e)
        ws2.cell(m+2,5,ch); ws2.cell(m+2,6,net); ws2.cell(m+2,7,cum)
        ws2.cell(m+2,8,round(random.uniform(0.32,0.46),2))
        for col in range(1,9): ws2.cell(m+2,col).border=bdr
    for i in range(1,9): ws2.column_dimensions[chr(64+i)].width=14
    # Sheet 3: Industry
    ws3 = wb.create_sheet("Industry")
    hdrs3 = ["Industry","Count","ARR","Avg","Retention","Growth","NPS"]
    for i,h in enumerate(hdrs3,1):
        c = ws3.cell(1,i,h); c.font=hfont; c.fill=hfill; c.border=bdr
    idata = [("Financial",85,18500000,217647,.94,.52,4.7),
             ("Healthcare",62,11200000,180645,.96,.48,4.8),
             ("Manufacturing",48,8900000,185417,.91,.35,4.5),
             ("Technology",78,14800000,189744,.89,.42,4.6),
             ("Retail",35,5600000,160000,.88,.38,4.4),
             ("Government",28,7200000,257143,.97,.30,4.9)]
    for r,d in enumerate(idata,2):
        for c,v in enumerate(d,1):
            ws3.cell(r,c,v); ws3.cell(r,c).border=bdr
    for i in range(1,8): ws3.column_dimensions[chr(64+i)].width=16
    wb.save(fp)
    print("  customer_sales_data.xlsx:", os.path.getsize(fp), "bytes")

# ===== Excel 2: Feature Comparison =====
def xlsx_features():
    fp = os.path.join(D, "feature_comparison_matrix.xlsx")
    wb = Workbook()
    hfont = Font(bold=True, color="FFFFFF"); hfill = PatternFill("solid", fgColor="548235")
    bdr = Border(*(Side(style="thin"),)*4)
    ws = wb.active; ws.title = "Features"
    hdrs = ["Category","Feature","V1.0","V2.0","V3.0","Priority","Weeks","Status"]
    for i,h in enumerate(hdrs,1):
        c = ws.cell(1,i,h); c.font=hfont; c.fill=hfill; c.border=bdr
    feats = [
        ("Doc Processing","PDF Parse","Basic","Advanced","OCR+Table","P0",8,"Done"),
        ("","DOCX","Text","Tables","Track Changes","P1",6,"Planned"),
        ("","XLSX","Cells","Formulas","Pivot","P2",10,"Planned"),
        ("","Image","N/A","OCR","GPT-4V","P1",12,"WIP"),
        ("","Video","N/A","N/A","Transcription","P3",16,"Research"),
        ("Retrieval","Vector","IVF","HNSW","Hybrid ANN","P0",6,"Done"),
        ("","Keyword","BM25","BM25+TF","Learned Sparse","P1",8,"WIP"),
        ("","Hybrid","N/A","Basic","RRF","P0",4,"Done"),
        ("","Rerank","N/A","CrossEnc","ColBERT+LLM","P1",10,"Planned"),
        ("","QueryExpand","N/A","Synonyms","LLM-based","P2",8,"Planned"),
        ("Generation","Q&A","Yes","Yes","Yes","P0",0,"Done"),
        ("","Streaming","N/A","Yes","Yes","P0",0,"Done"),
        ("","Citation","Page","Paragraph","Sentence","P1",6,"WIP"),
        ("","Multi-turn","N/A","Basic","Advanced","P1",12,"Planned"),
        ("","HallucDetect","N/A","N/A","Confidence","P1",10,"Research"),
        ("Security","Auth","API Key","JWT","SSO SAML","P0",8,"WIP"),
        ("","Access","N/A","RBAC","Doc-level","P0",12,"Planned"),
        ("","Encrypt","TLS1.2","TLS1.3","E2E","P0",6,"WIP"),
        ("","Audit","Basic","Detail","Compliance","P1",4,"Done"),
        ("","DataRes","N/A","Region","Multi-region","P2",16,"Planned"),
        ("Analytics","QueryLog","Basic","Detail","Full","P1",8,"WIP"),
        ("","Usage","N/A","Dashboard","Advanced","P2",10,"Planned"),
        ("","ROI","N/A","N/A","Built-in","P3",6,"Planned"),
        ("Integration","REST","Yes","Yes","Yes","P0",0,"Done"),
        ("","SDK-Py","N/A","Yes","Yes","P1",0,"Done"),
        ("","SDK-JS","N/A","N/A","Full","P2",8,"Planned"),
        ("","Plugins","N/A","N/A","Marketplace","P2",16,"Planned"),
    ]
    for r,d in enumerate(feats,2):
        for c,v in enumerate(d,1):
            ws.cell(r,c,v); ws.cell(r,c).border=bdr
    for i,w in enumerate([18,18,12,12,14,10,10,10]):
        ws.column_dimensions[chr(65+i)].width = w
    # Sheet 2: Competitor
    ws2 = wb.create_sheet("Competitor")
    hdrs2 = ["Capability","Ours","CompA","CompB","CompC","Market"]
    for i,h in enumerate(hdrs2,1):
        c = ws2.cell(1,i,h); c.font=hfont; c.fill=hfill; c.border=bdr
    comps = [("Format",9,7,8,6,7.5),("Accuracy",9,8,7,8,8.0),
             ("Latency",8,7,9,7,7.8),("Scale",9,8,7,8,8.0),
             ("Security",8,9,7,7,7.8),("Integration",8,7,8,6,7.2),
             ("Pricing",7,6,8,7,7.0),("Support",9,7,6,8,7.5),
             ("Docs",8,8,7,6,7.2),("Custom",9,6,7,8,7.5),
             ("AI/ML",9,9,7,7,8.0),("Overall",8.4,7.4,7.4,7.1,7.5)]
    for r,d in enumerate(comps,2):
        for c,v in enumerate(d,1):
            ws2.cell(r,c,v); ws2.cell(r,c).border=bdr
    for i in range(1,7): ws2.column_dimensions[chr(64+i)].width=14
    wb.save(fp)
    print("  feature_comparison_matrix.xlsx:", os.path.getsize(fp), "bytes")

# ===== Excel 3: Tech Metrics =====
def xlsx_metrics():
    fp = os.path.join(D, "technical_metrics_dashboard.xlsx")
    wb = Workbook()
    hfont = Font(bold=True, color="FFFFFF"); hfill = PatternFill("solid", fgColor="4472C4")
    bdr = Border(*(Side(style="thin"),)*4)
    # Sheet 1: Performance
    ws = wb.active; ws.title = "Performance"
    hdrs = ["Metric","P50","P90","P95","P99","Target","Status"]
    for i,h in enumerate(hdrs,1):
        c = ws.cell(1,i,h); c.font=hfont; c.fill=hfill; c.border=bdr
    perf = [("Query Latency(ms)",120,280,350,800,500,"Met"),
            ("Embedding(ms)",45,85,120,250,200,"Met"),
            ("DocProc(docs/min)",850,1200,1500,2000,1000,"Met"),
            ("IndexUpdate(ms)",150,320,480,1200,1000,"Met"),
            ("API Resp(ms)",95,180,240,520,300,"Met"),
            ("FirstToken(ms)",180,350,480,900,500,"Met"),
            ("ConcurrentUsers",850,1200,1400,1600,1000,"Met"),
            ("QPS",8500,11000,12500,14000,10000,"Met"),
            ("Mem(GB)",28,32,35,42,48,"Met"),
            ("CPU(%)",55,68,75,88,80,"Warn"),
            ("DiskIO(MB/s)",120,185,220,350,400,"Met"),
            ("NetLatency(ms)",5,12,18,35,50,"Met")]
    for r,d in enumerate(perf,2):
        for c,v in enumerate(d,1):
            ws.cell(r,c,v); ws.cell(r,c).border=bdr
    for i,w in enumerate([22,8,8,8,8,8,8]):
        ws.column_dimensions[chr(65+i)].width=w
    # Sheet 2: Daily
    ws2 = wb.create_sheet("Daily")
    hdrs2 = ["Date","Queries","Latency","ErrRate","NewDocs","Users","Revenue"]
    for i,h in enumerate(hdrs2,1):
        c = ws2.cell(1,i,h); c.font=hfont; c.fill=hfill; c.border=bdr
    for d in range(1,31):
        ws2.cell(d+1,1,f"2025-09-{d:02d}")
        ws2.cell(d+1,2,85000+random.randint(-10000,15000))
        ws2.cell(d+1,3,round(random.uniform(150,220),1))
        ws2.cell(d+1,4,round(random.uniform(0.01,0.05),3))
        ws2.cell(d+1,5,random.randint(500,2000))
        ws2.cell(d+1,6,random.randint(800,1200))
        ws2.cell(d+1,7,round(random.uniform(15000,25000),2))
        for col in range(1,8): ws2.cell(d+1,col).border=bdr
    for i in range(1,8): ws2.column_dimensions[chr(64+i)].width=12
    # Sheet 3: Errors
    ws3 = wb.create_sheet("Errors")
    hdrs3 = ["Type","Count","Pct","Resolution","Impact","Cause"]
    for i,h in enumerate(hdrs3,1):
        c = ws3.cell(1,i,h); c.font=hfont; c.fill=hfill; c.border=bdr
    errs = [("Timeout",1250,0.35,"2min","High","Large docs"),
            ("EmbedAPI",890,0.25,"5min","High","Rate limit"),
            ("Memory",450,0.13,"15min","Critical","OOM"),
            ("Connection",380,0.11,"1min","Medium","Network"),
            ("Parse",320,0.09,"30min","Medium","Format"),
            ("Index",150,0.04,"10min","High","Corrupted"),
            ("Auth",85,0.02,"0min","Low","Token"),
            ("Validation",45,0.01,"0min","Low","Params")]
    for r,d in enumerate(errs,2):
        for c,v in enumerate(d,1):
            ws3.cell(r,c,v); ws3.cell(r,c).border=bdr
    for i,w in enumerate([14,10,8,14,10,14]):
        ws3.column_dimensions[chr(65+i)].width=w
    wb.save(fp)
    print("  technical_metrics_dashboard.xlsx:", os.path.getsize(fp), "bytes")

# ===== CSV: Query Logs =====
def csv_logs():
    fp = os.path.join(D, "query_logs_sample.csv")
    queries = ["Revenue growth?","Q3 financial summary","Key risks?",
               "Customer satisfaction","Regional comparison","Retention rate?",
               "API error codes","RAG architecture","Security compliance",
               "2026 roadmap","Avg contract value","Market analysis",
               "Tech trends","Vendor comparison","Deployment checklist"]
    sources = ["financial_report_2025.xlsx","market_analysis_2025.pdf",
               "technical_architecture.md","customer_sales_data.xlsx",
               "enterprise_rag_whitepaper.pdf","product_roadmap_v3.pdf",
               "feature_comparison_matrix.xlsx","technical_metrics_dashboard.xlsx"]
    with open(fp,"w",encoding="utf-8") as f:
        f.write("log_id,timestamp,query,response_ms,status,confidence,sources,docs\n")
        for i in range(500):
            ts = f"2025-09-{random.randint(1,30):02d} {random.randint(8,20):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
            q = random.choice(queries); rt = random.randint(80,800)
            st = random.choice(["success","success","success","success","partial","error"])
            conf = round(random.uniform(0.7,0.99),3)
            sc = random.randint(1,5)
            docs = ";".join(random.sample(sources, min(sc,len(sources))))
            f.write(f"LOG-{20250901+i:08d},{ts},\"{q}\",{rt},{st},{conf},{sc},\"{docs}\"\n")
    print("  query_logs_sample.csv:", os.path.getsize(fp), "bytes")

# ===== CSV: A/B Test =====
def csv_abtest():
    fp = os.path.join(D, "ab_test_results.csv")
    with open(fp,"w",encoding="utf-8") as f:
        f.write("test_id,name,variant,impressions,clicks,conv,rate,conf,pval,lift\n")
        rows = [
            ("T001","UI Redesign","ctrl",50000,4500,850,0.017,0.95,0.023,0),
            ("T001","UI Redesign","varA",50000,5200,1020,0.0204,0.98,0.001,0.2),
            ("T002","Result Format","ctrl",45000,3800,720,0.016,0.90,0.045,0),
            ("T002","Result Format","varA",45000,4200,890,0.0198,0.96,0.008,0.24),
            ("T003","Ranking Algo","ctrl",60000,5400,1100,0.0183,0.85,0.082,0),
            ("T003","Ranking Algo","varA",60000,6100,1380,0.023,0.99,0.001,0.26),
            ("T004","Streaming","ctrl",55000,4800,960,0.0175,0.92,0.038,0),
            ("T004","Streaming","varA",55000,5500,1180,0.0215,0.97,0.003,0.23),
            ("T005","Upload Flow","ctrl",30000,2100,420,0.014,0.88,0.065,0),
            ("T005","Upload Flow","varA",30000,2400,520,0.0173,0.94,0.012,0.24),
        ]
        for r in rows:
            f.write(",".join(str(v) for v in r)+"\n")
    print("  ab_test_results.csv:", os.path.getsize(fp), "bytes")

print("=== Generating test files ===")
pdf_market()
pdf_whitepaper()
pdf_roadmap()
xlsx_customer()
xlsx_features()
xlsx_metrics()
csv_logs()
csv_abtest()
total = sum(os.path.getsize(os.path.join(D,f)) for f in os.listdir(D) if os.path.isfile(os.path.join(D,f)))
print(f"\nTotal dir size: {total/1024:.0f} KB")
