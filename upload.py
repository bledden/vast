#!/usr/bin/env python3
"""Push motion-event clips (written by the worker into events/) into VAST VSS, so they're
indexed by the VAST pipeline and searchable in its UI. Only events go in, never idle footage.

    VSS_USERNAME=team-47 python upload.py [--watch]    # prompts for the password

Each clip is uploaded once; a .uploaded marker records its object key.
"""
import argparse, getpass, glob, json, os, time, uuid, urllib.request

# Cloudflare in front of VSS rejects Python's default User-Agent (error 1010).
UA = "codec-vision/1.0"


def login(url, user, password):
    req = urllib.request.Request(f"{url}/api/v1/auth/login", data=json.dumps({"username": user, "password": password}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=30))["access_token"]


def upload(url, token, path, meta):
    boundary = uuid.uuid4().hex
    fields = {
        "is_public": "true",
        "tags": ",".join(["codec-vision", "motion-event", *meta.get("labels", [])]),
        "camera_id": meta.get("camera", "cam"),
        "custom_prompt": "Security camera clip, kept only because the camera's encoder detected motion. "
                         "Describe the people and vehicles, what each is doing, and where they go. "
                         f"A first look said: {meta.get('summary', '')}"[:800],
    }
    body = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode() for k, v in fields.items())
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
             "Content-Type: video/mp4\r\n\r\n").encode() + open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{url}/api/v1/videos/upload", data=body, headers={
        "Authorization": f"Bearer {token}", "Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=120))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true", help="keep uploading new events as they appear")
    args = p.parse_args()
    url = os.environ.get("VSS_URL", "https://team-47-vss.thecosmoslabs.com").rstrip("/")
    user = os.environ.get("VSS_USERNAME") or input("VSS username: ")
    token = login(url, user, os.environ.get("VSS_PASSWORD") or getpass.getpass(f"VSS password for {user}: "))
    print("logged in; uploading motion events from events/")
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "events")
    while True:
        for clip in sorted(glob.glob(os.path.join(folder, "*.mp4"))):
            base = clip[:-4]
            if os.path.exists(base + ".uploaded") or not os.path.exists(base + ".json"):
                continue
            meta = json.load(open(base + ".json"))
            res = upload(url, token, clip, meta)
            open(base + ".uploaded", "w").write(res.get("object_key", ""))
            print(f"uploaded {os.path.basename(clip)} -> {res.get('object_key')}: {meta.get('summary', '')[:80]}")
        if not args.watch:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
