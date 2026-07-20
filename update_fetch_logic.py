import re

file_path = "/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/src/ofa_main.py"
with open(file_path, "r") as f:
    orig_text = f.read()

# Replace block definitions to include fetch_blocks
old_def = """    bash_blocks = re.findall(r"```(?:bash|sh|shell)\\n(.*?)\\n```", response_text, re.DOTALL)
    search_blocks = re.findall(r"```(?:search)\\n(.*?)\\n```", response_text, re.DOTALL)

    if not bash_blocks and not search_blocks:
        return None"""

new_def = """    bash_blocks = re.findall(r"```(?:bash|sh|shell)\\n(.*?)\\n```", response_text, re.DOTALL)
    search_blocks = re.findall(r"```(?:search)\\n(.*?)\\n```", response_text, re.DOTALL)
    fetch_blocks = re.findall(r"```(?:fetch)\\n(.*?)\\n```", response_text, re.DOTALL)

    if not bash_blocks and not search_blocks and not fetch_blocks:
        return None"""

text = orig_text.replace(old_def, new_def)

# Find the search block processing where it auto-fetches
old_search_logic = """                    # Auto-fetch the first result for better context
                    first_url = results[0]['href']
                    print(f"Fetching content from top result: {first_url}")
                    try:
                        import httpx
                        from lxml import html
                        # Use a realistic User-Agent to avoid soft bans
                        resp = httpx.get(first_url, timeout=5.0, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
                        tree = html.fromstring(resp.content)
                        # Remove script, style, header, footer elements before parsing
                        for bad in tree.xpath('//script|//style|//header|//footer|//nav|//aside'):
                            bad.getparent().remove(bad)
                        
                        elements = tree.xpath('//text()')
                        cleaned = " ".join([t.strip() for t in elements if t.strip() and len(t.strip()) > 3])
                        
                        # Fallback for empty results
                        if not cleaned:
                            cleaned = "Unable to read dynamic webpage content cleanly, consider using a different tool or command."
                        out_str += f"\\n--- First Link Content Extract ---\\n{cleaned[:16000]}\\n----------------------------------\\n"
                    except Exception as e:
                        out_str += f"\\n--- First Link Fetch Failed ---\\n{str(e)}\\n----------------------------------\\n"
                        pass"""

# The new fetch processing block to be injected after the search logic
fetch_logic = """
    # Process fetch blocks
    for url in fetch_blocks:
        url = url.strip()
        if not url:
            continue
        print(f"\\n[Web Fetch Suggested]")
        print(f"URL: {url}")
        ans = input("Execute this fetch? [y/N]: ").strip().lower()
        if ans in ('y', 'yes'):
            print("-" * 60)
            try:
                import httpx
                from lxml import html
                resp = httpx.get(url, timeout=5.0, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
                tree = html.fromstring(resp.content)
                for bad in tree.xpath('//script|//style|//header|//footer|//nav|//aside'):
                    bad.getparent().remove(bad)
                
                elements = tree.xpath('//text()')
                cleaned = " ".join([t.strip() for t in elements if t.strip() and len(t.strip()) > 3])
                
                if not cleaned:
                    cleaned = "Unable to read dynamic webpage content cleanly."
                out_str = f"\\n--- Fetched URL: {url} ---\\n{cleaned[:16000]}\\n----------------------------------\\n"
                print(out_str)
                all_outputs.append(out_str)
            except Exception as e:
                err_msg = f"\\n--- Fetch Failed ---\\n{str(e)}\\n----------------------------------\\n"
                print(err_msg)
                all_outputs.append(err_msg)
            print("-" * 60)
"""

text = text.replace(old_search_logic, "")
text = text.replace("    # Process bash blocks", fetch_logic + "\n    # Process bash blocks")

with open(file_path, "w") as f:
    f.write(text)

print("Updated parser logic successfully.")
