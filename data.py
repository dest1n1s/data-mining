import functools
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests


def parse_markdown_file(file_content: str, date: str) -> list[dict[str, Any]]:
    # parse the markdown file
    lines = file_content.strip().split("\n")
    data = []
    # Get trending items
    item_pattern = re.compile(r"(\d+)\.\s+\[(.+?)\]\((.+?)\)")
    for line in lines:
        item_match = item_pattern.match(line)
        if item_match:
            data.append(
                {
                    "ranking": len(data) + 1,
                    "title": item_match.group(2),
                    "url": item_match.group(3),
                    "date": date,
                }
            )

    return data


def disk_cache(cache_dir: str = "cache"):
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(date: datetime, *args, **kwargs):
            date_str = date.strftime("%Y-%m-%d")
            cache_file = cache_path / f"{date_str}.json"

            if cache_file.exists():
                return json.loads(cache_file.read_text())

            result = func(date, *args, **kwargs)
            cache_file.write_text(json.dumps(result, ensure_ascii=False))
            return result

        return wrapper

    return decorator


@disk_cache()
def fetch_data_of_a_day(date: datetime, max_retries: int = 3) -> list[dict[str, Any]]:
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://cdn.jsdelivr.net/gh/justjavac/weibo-trending-hot-search@master/archives/{date_str}.md"

    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            result = parse_markdown_file(response.text, date_str)
            return result
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return []
            if attempt == max_retries - 1:  # Last attempt
                raise
            logging.warning(f"Failed to fetch data of {date_str}, retrying...")
            time.sleep(10)  # Wait 10 seconds before retrying
            continue
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:  # Last attempt
                raise
            logging.warning(f"Failed to fetch data of {date_str}, retrying...")
            time.sleep(10)  # Wait 10 seconds before retrying
            continue
    else:
        raise Exception("Failed to fetch data")


def fetch_all_data() -> list[dict[str, Any]]:
    # fetch data from 2020-11-24 to today
    data = []
    start_date = datetime(2020, 11, 24)
    end_date = datetime.now()
    current_date = start_date
    while current_date <= end_date:
        data.extend(fetch_data_of_a_day(current_date))
        current_date += timedelta(days=1)
    return data


if __name__ == "__main__":
    data = fetch_all_data()
    save_path = Path("data/data.jsonl")
    save_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in data)
    )
