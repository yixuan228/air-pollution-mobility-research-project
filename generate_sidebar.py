import yaml

# 1. 读取 toc.yml
with open("_toc.yml", "r", encoding="utf-8") as f:
    toc = yaml.safe_load(f)

# 2. 函数生成 sidebar 条目
def generate_sidebar(toc):
    lines = ["* [Home](README.md)\n"]  # 首页
    for part in toc.get("parts", []):
        caption = part.get("caption", "")
        if caption:
            lines.append(f"* {caption}")
        chapters = part.get("chapters", [])
        for chap in chapters:
            if "file" in chap:
                file_path = chap["file"]
                # 自动提取标题，如果有 title 用 title，否则用文件名
                title = chap.get("title", file_path.split("/")[-1].replace(".md", "").replace("_", " ").title())
                lines.append(f"  * [{title}]({file_path})")
            elif "url" in chap:
                url = chap["url"]
                title = chap.get("title", url)
                lines.append(f"  * [{title}]({url})")
        lines.append("")  # 空行分隔部分
    return "\n".join(lines)

# 3. 写入 _sidebar.md
sidebar_content = generate_sidebar(toc)
with open("_sidebar.md", "w", encoding="utf-8") as f:
    f.write(sidebar_content)

print("✅ _sidebar.md 已生成完成！")
