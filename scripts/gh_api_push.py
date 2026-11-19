# -*- coding: utf-8 -*-
"""GitHub Data API 推送 (当 github.com:443 被阻断、api.github.com 可用时)

用法: python gh_api_push.py <repo_path> <owner/repo> <base_commit_sha> "<commit message>"
"""
import subprocess, os, json, base64, sys, tempfile

GH = r"C:\Program Files\GitHub CLI\gh.exe"
BASH = r"C:\Users\22340\AppData\Local\hermes\tools\git-2.53.0+3-win32-x64\usr\bin\bash.exe"
GITBIN = r"C:\Users\22340\AppData\Local\hermes\tools\git-2.53.0+3-win32-x64\cmd"
ENV = {**os.environ, "PATH": GITBIN+";"+os.environ.get("PATH",""), "GIT_TERMINAL_PROMPT":"0"}

def sh(cmd, t=300):
    r = subprocess.run([BASH,"-lc",cmd], capture_output=True, timeout=t, env=ENV)
    return (r.stdout.decode("utf-8","replace")+r.stderr.decode("utf-8","replace")).strip()

def api(path, body=None, jq=None):
    args = f'"{GH}" api {path}'
    if body is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(body, f); tmp = f.name
        args += f' --method POST --input "{tmp}"'
    if jq: args += f' --jq "{jq}"'
    return sh(args)

def push(repo_path, repo, base_sha, message):
    # 1. 差异文件
    diff = sh(f'cd "{repo_path}" && git diff --name-status {base_sha} HEAD')
    changed = [l.split("\t") for l in diff.splitlines() if l.strip()]
    print(f"变更文件: {len(changed)}")

    # 2. blob + tree items
    items = []
    for status, path in changed:
        if status == "D":
            items.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            continue
        full = os.path.join(repo_path, path.replace("/", os.sep))
        b64 = base64.b64encode(open(full,"rb").read()).decode()
        sha = api(f"repos/{repo}/git/blobs", {"content": b64, "encoding": "base64"}, ".sha").strip()
        if len(sha) != 40:
            print(f"  ❌ {path}: {sha[:80]}"); continue
        items.append({"path": path, "mode": "100644", "type": "blob", "sha": sha})
        print(f"  ✓ {path}")

    # 3. tree
    full_base = api(f"repos/{repo}/commits/{base_sha}", jq=".sha").strip()
    tree = json.loads(api(f"repos/{repo}/git/trees", {"base_tree": full_base, "tree": items}))
    print(f"tree: {tree['sha'][:7]}")

    # 4. commit
    cm = json.loads(api(f"repos/{repo}/git/commits",
                        {"message": message, "tree": tree["sha"], "parents": [full_base]}))
    print(f"commit: {cm['sha'][:7]}")

    # 5. 更新分支
    ref = api(f"repos/{repo}/git/refs/heads/master", {"sha": cm["sha"], "force": False})
    try:
        print(f"✅ 分支已更新: {json.loads(ref)['object']['sha'][:7]}")
    except Exception:
        print(f"❌ {ref[:300]}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__); sys.exit(1)
    push(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
