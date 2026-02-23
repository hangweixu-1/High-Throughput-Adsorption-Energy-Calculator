import os, html
from .util import ensure_dir

def write_dashboard(outdir, ranked_rows, title="HTAEC Dashboard"):
    ensure_dir(outdir)
    top=ranked_rows[:200]
    headers=list(top[0].keys()) if top else []
    rows=[]
    for r in top:
        rows.append("<tr>"+"".join(f"<td>{html.escape(str(r.get(h,'')))}</td>" for h in headers)+"</tr>")
    doc=f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial; margin:20px;}}
table{{border-collapse:collapse; width:100%; font-size:12px;}}
th,td{{border:1px solid #ddd; padding:6px;}}
th{{position:sticky; top:0; background:#f7f7f7;}}
input{{padding:8px; width:420px; margin:10px 0;}}
</style>
</head>
<body>
<h2>{html.escape(title)}</h2>
<p>Top {len(top)} candidates (showing up to 200). Search to filter.</p>
<input id="q" placeholder="Search..." oninput="filterTable()"/>
<table id="t"><thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<script>
function filterTable(){{
  const q=document.getElementById('q').value.toLowerCase();
  const rows=document.querySelectorAll('#t tbody tr');
  rows.forEach(r=>{{ r.style.display = r.innerText.toLowerCase().includes(q)?'':'none'; }});
}}
</script>
</body></html>"""
    with open(os.path.join(outdir,"dashboard.html"),"w",encoding="utf-8") as f:
        f.write(doc)
