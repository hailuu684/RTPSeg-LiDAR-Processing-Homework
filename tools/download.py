"""Download a shared dataset archive: Google Drive (gdown, resumable) or a OneDrive share link (public shares API)."""
import argparse
import base64
import shutil
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--source', choices=['gdrive', 'onedrive'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--force', action='store_true', help='download again even if --output exists')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and args.output.stat().st_size > 0 and not args.force:
        print(f'{args.output} exists ({args.output.stat().st_size / 2 ** 30:.2f} GiB), skip download')
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.source == 'gdrive':
        import gdown
        gdown.download(args.url, str(args.output), quiet=False, fuzzy=True, resume=True)
    else:
        token = base64.urlsafe_b64encode(args.url.encode()).decode().rstrip('=')
        tmp = args.output.with_name(args.output.name + '.part')
        with urllib.request.urlopen(f'https://api.onedrive.com/v1.0/shares/u!{token}/root/content') as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f, length=16 * 2 ** 20)
        tmp.rename(args.output)
    print(f'saved {args.output} ({args.output.stat().st_size / 2 ** 30:.2f} GiB)')


if __name__ == '__main__':
    main()
