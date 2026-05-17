import requests
import re
import csv
import time
import random
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================================
# FIX UTF-8 CONSOLE WINDOWS
# =========================================
try:
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

BASE_URL = "https://diemthi.tuyensinh247.com"

TARGET_YEARS = [2025, 2024, 2023]

# Crawl full 8 methods
METHOD_IDS = [1]

# =========================================
# SESSION + RETRY
# =========================================
session = requests.Session()

retry = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry)

session.mount("http://", adapter)
session.mount("https://", adapter)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Connection": "keep-alive",
}

API_HEADERS = {
    **HEADERS,
    "Accept": "application/json",
}

session.headers.update(HEADERS)

# =========================================
# GET ALL SCHOOL LINKS
# =========================================
def get_all_school_links():

    url = BASE_URL + "/diem-chuan.html"

    try:
        res = session.get(url, timeout=15)

    except Exception as e:
        print("❌ Lỗi load danh sách trường:", e)
        return []

    pattern = r'<a[^>]*href="/diem-chuan/([^"]+)-([A-Z0-9]+)\.html"[^>]*>(.*?)</a>'

    schools = []

    for m in re.finditer(pattern, res.text):

        slug = m.group(1)
        code = m.group(2)
        raw_name = m.group(3)
        clean_name = re.sub(r'<[^>]+>', '', raw_name).strip()

        school_name = f"{code} - {clean_name}"

        school_url = (
            f"{BASE_URL}/diem-chuan/{slug}-{code}.html"
        )

        schools.append((school_name, school_url, code))

    # remove duplicate but keep order
    schools = list(dict.fromkeys(schools))

    return schools


# =========================================
# EXTRACT SCHOOL_ID
# =========================================
def get_school_id_from_html(html: str):

    patterns = [
        r'"school_id"\s*:\s*(\d+)',
        r'school_id\\?"?\s*:\s*(\d+)',
    ]

    for p in patterns:

        m = re.search(p, html)

        if m:
            return int(m.group(1))

    return None


# =========================================
# FETCH API
# =========================================
def fetch_api(school_id, method_id, year):

    url = (
        f"{BASE_URL}/api/common/cutoff-score"
        f"?school_id={school_id}"
        f"&method_id={method_id}"
        f"&year={year}"
    )

    try:

        r = session.get(
            url,
            headers=API_HEADERS,
            timeout=15
        )

        if r.status_code != 200:
            return []

        content_type = r.headers.get(
            "Content-Type", ""
        )

        if "application/json" not in content_type:
            return []

        json_data = r.json()

        if not isinstance(json_data, dict):
            return []

        data = json_data.get("data")

        if not data:
            return []

        return data

    except Exception as e:

        print(
            f"❌ API Error "
            f"(school_id={school_id}, "
            f"method={method_id}, "
            f"year={year})"
        )

        print(e)

        return []


# =========================================
# NORMALIZE
# =========================================
def normalize(rec, truong, year, method_id):

    ten_nganh = (
        rec.get("name")
        or rec.get("major_name")
        or rec.get("nganh")
        or rec.get("ten_nganh")
        or ""
    )

    to_hop = (
        rec.get("block")
        or rec.get("to_hop")
        or rec.get("combination")
        or ""
    )

    diem = (
        rec.get("mark")
        or rec.get("diem_chuan")
        or rec.get("score")
        or rec.get("cutoff_score")
        or ""
    )

    ma_nganh = (
        rec.get("code")
        or rec.get("major_code")
        or ""
    )

    return {
        "truong": truong,
        "ma_nganh": str(ma_nganh).strip(),
        "ten_nganh": str(ten_nganh).strip(),
        "to_hop": str(to_hop).strip(),
        "diem_chuan": str(diem).strip(),
        "nam": year,
        "method_id": method_id
    }


# =========================================
# MAIN
# =========================================
def main():

    all_data = []

    # chống duplicate
    seen = set()

    schools = get_all_school_links()

    print(f"\n✅ Tổng số trường: {len(schools)}\n")

    for idx, (name, link, code) in enumerate(schools, 1):

        print("=" * 70)
        print(f"[{idx}/{len(schools)}]")
        print(f"🏫 Trường: {name}")

        try:

            res = session.get(
                link,
                timeout=15
            )

        except Exception as e:

            print("❌ Lỗi tải HTML:", e)
            continue

        school_id = get_school_id_from_html(
            res.text
        )

        if not school_id:

            print(
                "❌ Không tìm thấy school_id"
            )

            continue

        print(f"🆔 school_id = {school_id}")

        total_school_rows = 0

        # =====================================
        # 1 TRƯỜNG
        # -> 8 METHOD
        # -> 3 NĂM
        # = 24 REQUEST
        # =====================================

        for method_id in METHOD_IDS:

            print(f"\n📌 METHOD {method_id}")

            for year in TARGET_YEARS:

                print(
                    f"   ↳ Năm {year} ... ",
                    end=""
                )

                records = fetch_api(
                    school_id,
                    method_id,
                    year
                )

                if records:

                    print(
                        f"✅ {len(records)} dòng"
                    )

                    for rec in records:

                        row = normalize(
                            rec,
                            name,
                            year,
                            method_id
                        )

                        # bỏ dữ liệu rỗng
                        if (
                            not row["ten_nganh"]
                            or not row["diem_chuan"]
                        ):
                            continue

                        # chống duplicate
                        key = (
                            row["truong"],
                            row["ma_nganh"],
                            row["ten_nganh"],
                            row["to_hop"],
                            row["diem_chuan"],
                            row["nam"],
                        )

                        if key not in seen:

                            seen.add(key)

                            all_data.append(row)

                            total_school_rows += 1

                else:
                    print("❌ Không có dữ liệu")

                # delay giữa request API
                time.sleep(
                    random.uniform(0.3, 0.8)
                )

        print(
            f"\n🎯 Tổng dòng trường này:"
            f" {total_school_rows}"
        )

        # delay giữa trường
        time.sleep(
            random.uniform(2, 4)
        )

        # autosave mỗi 20 trường
        if idx % 20 == 0:

            print("\n💾 Auto save...")

            with open(
                "diem_chuan_all.csv",
                "w",
                newline="",
                encoding="utf-8-sig"
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "truong",
                        "ma_nganh",
                        "ten_nganh",
                        "to_hop",
                        "diem_chuan",
                        "nam",
                        "method_id"
                    ]
                )

                writer.writeheader()
                writer.writerows(all_data)

    # =========================================
    # FINAL SAVE
    # =========================================
    output_file = "diem_chuan_all.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "truong",
                "ma_nganh",
                "ten_nganh",
                "to_hop",
                "diem_chuan",
                "nam",
                "method_id"
            ]
        )

        writer.writeheader()
        writer.writerows(all_data)

    print("\n" + "=" * 70)

    print("✅ DONE!")

    print(f"📄 File: {output_file}")

    print(f"📊 Tổng số dòng: {len(all_data)}")

    print("=" * 70)


if __name__ == "__main__":
    main()