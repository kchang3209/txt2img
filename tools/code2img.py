import asyncio
import csv
import json
import os
from jinja2 import Template
from playwright.async_api import async_playwright
import hashlib
import argparse



# ================= HTML TEMPLATE =================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\">

<link rel=\"stylesheet\"
 href=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/{{ theme }}.min.css\">

<script src=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js\"></script>

<style>
* { box-sizing: border-box; }

html, body {
    margin: 0;
    padding: 0;
    background: {{ background }};
}

.container {
    width: {{ width }}px;
    height: auto;
    padding: {{ padding }}px;
    overflow: hidden;
}

pre, code, .hljs {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100%;
    width: 100%;
}

code {
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: anywhere;
    font-size: {{ font_size }}px;
    line-height: {{ line_height }};
    font-family: \"JetBrains Mono\", \"Fira Code\", monospace;
}

body, .container, pre, code, .hljs {
    background: #ffffff !important;
}
</style>
</head>

<body>
<div class=\"container\">
<pre><code class=\"language-plaintext\">{{ code }}</code></pre>
</div>

<script>
hljs.highlightAll();
</script>
</body>
</html>
"""
# <pre><code class=\"language-python\">{{ code }}</code></pre>  # code w/ highlight

# ================= UTILS =================
def escape_html(s: str):
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )

async def render_prompt_image(prompt: str, task_id: str):
    h = hashlib.md5(prompt.encode()).hexdigest()[:12]
    img_name = f"{task_id}_{h}.png"
    out_path = os.path.join(IMAGE_DIR, img_name)

    html = Template(HTML_TEMPLATE).render(
        code=escape_html(prompt),
        width=IMG_WIDTH,
        # height=IMG_HEIGHT,
        padding=PADDING,
        font_size=FONT_SIZE,
        line_height=LINE_HEIGHT,
        theme=THEME,
        background=BACKGROUND
    )

    html_path = "/content/temp_render.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--single-process",
                "--no-zygote"
            ]
        )

        context = await browser.new_context(
            viewport={"width": IMG_WIDTH, "height": 1000},
            device_scale_factor=1
        )

        page = await context.new_page()
        await page.goto(f"file://{html_path}", wait_until="load")
        await page.wait_for_timeout(500)


        # ---------- GET ACTUAL CONTENT HEIGHT ----------
        page_height = await page.evaluate("""
            () => {
                const body = document.body;
                const html = document.documentElement;
                return Math.max(
                    body.scrollHeight,
                    body.offsetHeight,
                    html.clientHeight,
                    html.scrollHeight,
                    html.offsetHeight
                );
            }
        """)

        # ---------- RESIZE VIEWPORT ----------
        await page.set_viewport_size({
            "width": IMG_WIDTH,
            "height": page_height
        })
        await page.wait_for_timeout(200)

        # ---------- SCREENSHOT ----------
        await page.screenshot(path=out_path, full_page=True)

        await context.close()
        await browser.close()

    return out_path

async def process():
    text_dataset = []
    vlm_dataset = []
    with open(CSV_PATH, newline='', encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    for i, row in enumerate(rows):
        task_id = row["question_id"]  # e.g. leetcode_6
        prompt = row["prompt"]
        # canonical_solution = row["canonical_solution"]
        # test = row["test"]
        # entry_point = row["entry_point"]
        
        # -------- TEXT-ONLY FORMAT --------
        text_item = {
            "task_id": task_id,
            "prompt": prompt,
            # "canonical_solution": canonical_solution,
            # "test": test,
            # "entry_point": entry_point
        }
        text_dataset.append(text_item)
        
        # -------- VLM FORMAT --------
        img_path = await render_prompt_image(prompt, task_id)

        # `vlm_item` controls which columns in the csv to be included in the JSON file.
        vlm_item = {
            "task_id": task_id,
            "image": img_path,
            "prompt_image": os.path.basename(img_path)

        }
        vlm_dataset.append(vlm_item)
        if i % 10 == 0:
            print(f"Processed {i}/{len(rows)}")

    with open(os.path.join(TEXT_DATASET_DIR, "text.json"), "w", encoding="utf-8") as f:
        json.dump(text_dataset, f, indent=2)

    with open(os.path.join(VLM_DATASET_DIR, "vlm.json"), "w", encoding="utf-8") as f:
        json.dump(vlm_dataset, f, indent=2)

    print("\nDatasets created:")
    print("Text-only:", os.path.join(TEXT_DATASET_DIR, "text.json"))
    print("VLM:", os.path.join(VLM_DATASET_DIR, "vlm.json"))
    print("Images:", IMAGE_DIR)
 

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert code scripts into code images and JSON datasets."
    )

    parser.add_argument(
        "--csv_path",
        default=DEFAULT_CSV_PATH,
        help="Path to the code CSV file."
    )

    parser.add_argument(
        "--output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where datasets and images will be saved."
    )

    parser.add_argument(
        "--img_width",
        type=int,
        default=DEFAULT_IMG_WIDTH,
        help="Image width in pixels."
    )

    # parser.add_argument(
    #     "--img_height",
    #     type=int,
    #     default=DEFAULT_IMG_HEIGHT,
    #     help="Image height in pixels."
    # )

    parser.add_argument(
        "--font_size",
        type=int,
        default=DEFAULT_FONT_SIZE,
        help="Font size."
    )

    parser.add_argument(
        "--line_height",
        type=float,
        default=DEFAULT_LINE_HEIGHT,
        help="Line height."
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=DEFAULT_PADDING,
        help="Image padding."
    )

    parser.add_argument(
        "--theme",
        default=DEFAULT_THEME,
        help="highlight.js theme."
    )

    parser.add_argument(
        "--background",
        default=DEFAULT_BACKGROUND,
        help="Background color."
    )

    return parser.parse_args()

# ================= CONFIG =================
DEFAULT_CSV_PATH = ""
DEFAULT_OUTPUT_ROOT = "code2img"
DEFAULT_IMG_WIDTH = 1280
# DEFAULT_IMG_HEIGHT = 1024
DEFAULT_FONT_SIZE = 16
DEFAULT_LINE_HEIGHT = 1.4
DEFAULT_PADDING = 0
DEFAULT_THEME = "atom-one-light"
DEFAULT_BACKGROUND = "#ffffff"

args = parse_args()

CSV_PATH = args.csv_path
OUTPUT_ROOT = args.output_root

TEXT_DATASET_DIR = os.path.join(OUTPUT_ROOT, "text_only")
VLM_DATASET_DIR = os.path.join(OUTPUT_ROOT, "vlm")
IMAGE_DIR = os.path.join(VLM_DATASET_DIR, "images")

IMG_WIDTH = args.img_width
# IMG_HEIGHT = args.img_height
FONT_SIZE = args.font_size
LINE_HEIGHT = args.line_height
PADDING = args.padding
THEME = args.theme
BACKGROUND = args.background

os.makedirs(TEXT_DATASET_DIR, exist_ok=True)
os.makedirs(VLM_DATASET_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# ================= MAIN =================
async def main():
    await process()

if __name__ == "__main__":
    asyncio.run(main())