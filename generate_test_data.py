"""Generate complete test dataset with PDFs."""
import os

output_dir = "D:\\Desktop\\RAG\\test_data"
os.makedirs(output_dir, exist_ok=True)

# All document contents
docs = {
    "company_overview.txt": "XX科技有限公司简介\n\n成立于2020年，专注AI和大数据技术。总部北京，在上海、深圳、成都设有分支。\n核心业务：企业级RAG系统、智能客服、知识图谱、NLP咨询。\n员工500+，研发占60%。2025年营收5亿，增长45%。",
    
    "hr_policy.docx": "员工手册\n\n入职：提供证件，签3年合同，1周培训+导师辅导。\n试用期3个月，工资80%。\n工作时间：9:00-18:00，弹性核心10:00-16:00。\n年假：满1年5天，满10年10天。\n福利：五险一金最高基数，餐补500/月，交通300/月。",
    
    "technical_architecture.md": "# 技术架构\n\n微服务架构：Kong网关 + Go/Python服务。\n数据层：PostgreSQL + Milvus + Redis + MinIO + Kafka。\n部署：Kubernetes，3 Master + 10 Worker。\n监控：Prometheus + Grafana + ELK + Jaeger。",
    
    "financial_report_2025.xlsx": "2025财务报告\n\n营收5.2亿(+45%)，Q1:0.8亿 Q2:1.2亿 Q3:1.5亿 Q4:1.7亿\n成本：人力40%，研发29%，市场15%，运营10%\n毛利65%，净利18%，净利润0.94亿\n客户500+，续约率92%",
    
    "product_manual.pdf": "智能文档助手V3.0手册\n\n基于AI的企业级文档管理问答系统。\n功能：多格式解析、智能问答、知识图谱。\n支持PDF/DOCX/XLSX/MD/TXT，单文件50MB。\n检索准确率>95%，响应<3秒。",
    
    "meeting_minutes_20250901.txt": "Q3产品规划会议\n时间：2025-09-01 14:00\n参会：张总、李总、王总监、陈经理、赵工\n\nQ2：RAG V2.0上线，满意度95%，新增客户30家。\nQ3目标：V3.0开发，多语言支持，检索准确率98%。\n需招5名高级工程师，服务器预算增50万。",
    
    "api_documentation.md": "# API文档\n\nPOST /auth/login - 登录获取Token\nPOST /query - RAG查询\nPOST /documents/upload - 上传文档\nGET /documents - 文档列表\n\n限流：用户100次/分，全局10000次/分。",
    
    "security_policy.docx": "信息安全管理规范\n\n密码：12位+，含大小写数字特殊字符，90天更换。\n多因素认证必须启用。\n数据分类：公开/内部/机密/绝密。\n传输TLS1.3，存储AES-256。\n备份：日增量+周全量，异地存储。",
    
    "rag_system_spec.md": "# RAG技术规格\n\n检索延迟P95<500ms，准确率Recall@10>90%。\n生成：首Token<1s，速度>30 tokens/s。\n容量：100万文档，1000 QPS，99.9%可用。\n向量维度3072，混合检索Milvus+BM25。",
    
    "case_study_bank.txt": "银行智能客服案例\n\n某银行，总资产5万亿，员工3万。\n挑战：日均10万咨询，5万份文档，培训3-6月。\n方案：部署智能文档助手V3.0。\n效果：响应从5分钟降至30秒，解决率70%→95%，满意度82%→96%，年省2000万。",
    
    "project_plan_q3.txt": "Q3项目计划\n\n目标：V3.0版本开发\n里程碑：\n9/10 核心功能完成\n9/20 测试完成\n9/30 正式发布",
    
    "technical_faq.md": "# 技术FAQ\n\nQ: Milvus连接配置？\nA: .env设置MILVUS_HOST和MILVUS_PORT\n\nQ: 支持格式？\nA: PDF/DOCX/XLSX/MD/TXT",
    
    "onboarding_guide.txt": "新员工入职指南\n\n第一天：领工卡电脑，开邮箱账号，入职培训\n第一周：了解团队项目，配置环境，完成首个任务",
    
    "api_rate_limits.txt": "API限流配置\n\n用户级：查询100次/分，上传10次/时\n全局：10000次/分",
    
    "deployment_checklist.md": "# 部署检查清单\n\n部署前：环境变量、数据库迁移、SSL证书\n部署后：健康检查、监控告警、回归测试",
    
    "performance_report.txt": "性能测试报告\n\n环境：3台4核8G\nQPS：850\nP99延迟：1.2秒\n错误率：0.01%\nCPU：65%，内存：72%",
    
    "vendor_comparison.md": "# 供应商对比\n\n|维度|A|B|C|\n|价格|高|中|低|\n|性能|优|良|中|\n|服务|好|好|一般|",
    
    "compliance_checklist.txt": "合规检查\n\nGDPR：数据加密[✓] 用户同意[✓] 数据最小化[✓] 删除机制[ ]\n安全：访问控制[✓] 审计日志[✓] 漏洞扫描[✓]",
    
    "incident_report_0915.txt": "故障报告 2025-09-15\n\n时间：14:30-15:45\n影响：查询服务不可用\n根因：Milvus节点OOM\n处理：扩容内存64G，优化超时，增加监控",
    
    "meeting_notes_product.txt": "产品评审会议\n\nV3.0功能：多语言支持[通过] 知识图谱[延期V3.1] 流式输出[通过]",
    
    "training_material.txt": "技术培训材料\n\n第一课：系统架构（微服务、核心组件）\n第二课：开发规范（代码规范、Git工作流）\n第三课：部署流程（Docker、K8s）",
    
    "security_audit_2025.txt": "2025安全审计\n\n时间：2025年8月\n高风险：0，中风险：2，低风险：5\n建议：更新依赖，加强日志监控",
    
    "data_retention_policy.txt": "数据保留策略\n\n用户数据：账号永久，操作日志1年，查询6个月\n文档数据：原始永久，索引随文档，缓存24小时",
    
    "disaster_recovery_plan.md": "# 灾难恢复\n\nRTO：4小时，RPO：1小时\n步骤：评估→备份系统→数据恢复→验证→切换",
    
    "changelog_v3.0.md": "# Changelog v3.0\n\nAdded：多语言、流式输出、查询历史\nImproved：准确率98%，响应优化50%\nFixed：PDF特殊字符解析",
    
    # Additional docs to reach ~50
    "project_status_week36.txt": "第36周项目状态\n\n后端：V3.0核心功能90%完成\n前端：问答界面开发中\n测试：单元测试覆盖85%\n\n风险：依赖库版本兼容性问题",
    
    "architecture_decision_001.md": "# ADR-001: 选择Milvus作为向量数据库\n\n状态：已采纳\n\n决策：采用Milvus 2.4\n理由：开源、高性能、云原生、社区活跃\n备选：Pinecone(付费)、Weaviate、Qdrant",
    
    "bug_report_1024.txt": "Bug #1024\n\n标题：PDF解析中文乱码\n优先级：高\n描述：部分PDF文件中文显示为方块\n复现：上传含中文的PDF\n原因：编码检测失败\n修复：增加编码fallback链",
    
    "customer_feedback_sept.txt": "9月客户反馈汇总\n\n正面：\n- 检索速度快\n- 答案准确\n- 界面友好\n\n建议：\n- 增加导出功能\n- 支持更多格式\n- 优化移动端体验",
    
    "infrastructure_cost.txt": "基础设施成本分析\n\n月度成本：\n- 服务器：15万\n- 云服务：8万\n- 数据库：3万\n- AI API：5万\n总计：31万/月\n\n优化建议：\n- 使用预留实例\n- 优化缓存策略\n- 减少API调用",
    
    "team_structure.md": "# 团队结构\n\n技术部(200人)：\n- 后端组：60人\n- 前端组：40人\n- AI组：50人\n- 测试组：30人\n- 运维组：20人\n\n产品部(30人)：\n- 产品经理：10人\n- 设计师：10人\n- 运营：10人",
    
    "tech_stack_summary.txt": "技术栈总结\n\n后端：Python/FastAPI, Go, Node.js\n前端：React, TypeScript, TailwindCSS\n数据库：PostgreSQL, Redis, Milvus, MinIO\n基础设施：Docker, K8s, Prometheus, Grafana\nAI：OpenAI, DeepSeek, sentence-transformers",
    
    "quarterly_okr_q3.txt": "Q3 OKR\n\nO1: 提升产品竞争力\nKR1: V3.0按时发布\nKR2: 检索准确率98%\nKR3: 客户NPS>50\n\nO2: 扩大市场份额\nKR1: 新增客户50家\nKR2: 合同金额1亿\nKR3: 品牌知名度提升30%",
    
    "risk_assessment_2025.txt": "2025风险评估\n\n高风险：\n- 核心人员流失\n- 技术债务累积\n\n中风险：\n- 竞品威胁\n- 供应链中断\n\n低风险：\n- 汇率波动\n- 政策变化",
    
    "vendor_contract_template.txt": "供应商合同模板\n\n甲方：XX科技\n乙方：供应商\n\n条款：\n1. 服务范围：明确交付物\n2. 付款条件：30%预付，70%验收后\n3. 保密条款：3年有效期\n4. 违约责任：日0.5%违约金",
    
    "system_requirements.txt": "系统要求\n\n服务器：\n- CPU：8核+\n- 内存：32GB+\n- 存储：500GB SSD\n- 网络：1Gbps\n\n软件：\n- Python 3.11+\n- Docker 24.0+\n- K8s 1.28+",
    
    "meeting_notes_investor.txt": "投资人沟通会\n\n时间：2025-09-20\n参会：CEO, CFO, 投资人A/B\n\n议题：\n1. 业务进展\n2. 财务数据\n3. 下一轮融资\n\n结论：投资人认可进展，期待Q4数据",
    
    "product_roadmap_2025.md": "# 2025产品路线图\n\nQ1：V2.0发布\nQ2：客户增长\nQ3：V3.0多语言\nQ4：企业版、知识图谱\n\n2026：国际化、AI Agent",
    
    "performance_optimization.txt": "性能优化方案\n\n1. 缓存优化：L1内存+L2 Redis\n2. 数据库：读写分离，连接池\n3. 检索：IVF_FLAT索引，nlist=1024\n4. 生成：流式输出，异步处理\n5. 网络：CDN加速，压缩传输",
    
    "api_error_codes.txt": "API错误码\n\n400: 请求参数错误\n401: 未认证\n403: 无权限\n404: 资源不存在\n413: 文件过大\n429: 请求过于频繁\n500: 服务器内部错误\n503: 服务不可用",
    
    "data_schema_v2.txt": "数据模型 V2\n\nusers: id, username, email, role, department\ndocuments: id, title, file_type, status, owner_id\nchunks: id, doc_id, content, embedding_id\nmetadata: id, doc_id, key, value, source\nquery_logs: id, user_id, query, response, latency",
}

# Generate all files
print("Generating %d documents..." % len(docs))

# Generate TXT files
for name, content in docs.items():
    if name.endswith(".txt"):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            f.write(content)

# Generate MD files
for name, content in docs.items():
    if name.endswith(".md"):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            f.write(content)

# Generate DOCX files
try:
    from docx import Document
    for name, content in docs.items():
        if name.endswith(".docx"):
            d = Document()
            d.add_heading(name.replace(".docx", "").replace("_", " ").title(), 0)
            for line in content.split("\n"):
                if line.strip():
                    d.add_paragraph(line.strip())
            d.save(os.path.join(output_dir, name))
    print("DOCX files created")
except:
    print("DOCX skipped")

# Generate XLSX files
try:
    from openpyxl import Workbook
    for name, content in docs.items():
        if name.endswith(".xlsx"):
            wb = Workbook()
            ws = wb.active
            for line in content.split("\n"):
                if line.strip():
                    ws.append([line.strip()])
            wb.save(os.path.join(output_dir, name))
    print("XLSX files created")
except:
    print("XLSX skipped")

# Generate PDF files
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    
    for name, content in docs.items():
        if name.endswith(".pdf"):
            filepath = os.path.join(output_dir, name)
            c = canvas.Canvas(filepath, pagesize=A4)
            width, height = A4
            y = height - 2*cm
            for line in content.split("\n"):
                if y < 2*cm:
                    c.showPage()
                    y = height - 2*cm
                if line.strip():
                    # Use basic font for PDF
                    c.setFont("Helvetica", 10)
                    c.drawString(2*cm, y, line[:90])
                    y -= 0.5*cm
            c.save()
    print("PDF files created")
except Exception as e:
    print("PDF error:", e)

# Count files
txt = len([f for f in os.listdir(output_dir) if f.endswith(".txt")])
md = len([f for f in os.listdir(output_dir) if f.endswith(".md")])
docx = len([f for f in os.listdir(output_dir) if f.endswith(".docx")])
xlsx = len([f for f in os.listdir(output_dir) if f.endswith(".xlsx")])
pdf = len([f for f in os.listdir(output_dir) if f.endswith(".pdf")])

print("\n=== Dataset Summary ===")
print("TXT:   %d files" % txt)
print("MD:    %d files" % md)
print("DOCX:  %d files" % docx)
print("XLSX:  %d files" % xlsx)
print("PDF:   %d files" % pdf)
print("TOTAL: %d files" % (txt + md + docx + xlsx + pdf))
print("\nLocation: %s" % output_dir)
