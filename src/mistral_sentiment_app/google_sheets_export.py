import json
import os
from base64 import b64decode
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials

DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID = "1WmjMDPnTO4ZyOLR0lfDQ8CB-yaRZEWj_Uv2Y58jQj8g"
DEFAULT_SUMMARY_WORKSHEET = "sentiment_summary"
DEFAULT_KEYWORDS_WORKSHEET = "keyword_mentions"
GOOGLE_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _parse_service_account_json(raw: str) -> dict:
    candidates = [raw.strip()]
    stripped = raw.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        candidates.append(stripped[1:-1])

    parse_errors: list[str] = []

    for candidate in candidates:
        for payload in (candidate,):
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                if isinstance(parsed, dict):
                    private_key = parsed.get("private_key")
                    if isinstance(private_key, str):
                        parsed["private_key"] = private_key.replace("\\n", "\n")
                    return parsed
            except Exception as exc:  # noqa: BLE001
                parse_errors.append(str(exc))

        try:
            decoded = b64decode(candidate, validate=True).decode("utf-8")
            parsed = json.loads(decoded)
            if isinstance(parsed, dict):
                private_key = parsed.get("private_key")
                if isinstance(private_key, str):
                    parsed["private_key"] = private_key.replace("\\n", "\n")
                return parsed
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(str(exc))

    raise RuntimeError(
        "GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON is not valid JSON (or base64-encoded JSON). "
        "Ensure the env var contains the full service account document. "
        f"Parse attempts failed: {parse_errors[:2]}"
    )


def _get_required_credentials() -> Credentials:
    credentials_file = os.getenv("GOOGLE_SHEETS_SERVICE_ACCOUNT_FILE", "").strip()
    credentials_json = os.getenv("GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON", "").strip()

    if credentials_file:
        return Credentials.from_service_account_file(credentials_file, scopes=GOOGLE_SHEETS_SCOPES)

    if credentials_json:
        return Credentials.from_service_account_info(
            _parse_service_account_json(credentials_json),
            scopes=GOOGLE_SHEETS_SCOPES,
        )

    raise RuntimeError(
        "Google Sheets export requires GOOGLE_SHEETS_SERVICE_ACCOUNT_FILE or GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON"
    )


def _get_client() -> gspread.Client:
    return gspread.authorize(_get_required_credentials())


def _get_or_create_worksheet(
    spreadsheet: gspread.Spreadsheet,
    title: str,
    headers: list[str],
) -> gspread.Worksheet:
    try:
        worksheet = spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=title, rows=1000, cols=max(26, len(headers)))
        worksheet.append_row(headers, value_input_option="RAW")
        return worksheet

    if worksheet.row_count == 0:
        worksheet.append_row(headers, value_input_option="RAW")
        return worksheet

    existing_headers = worksheet.row_values(1)
    if not existing_headers:
        worksheet.append_row(headers, value_input_option="RAW")

    return worksheet


def write_results_to_google_sheets(
    result: dict,
    spreadsheet_id: str,
    summary_worksheet_name: str,
    keywords_worksheet_name: str,
) -> dict:
    client = _get_client()
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
    except PermissionError as exc:
        service_account_email = getattr(client.auth, "service_account_email", "unknown")
        raise RuntimeError(
            "Google Sheets access denied while opening spreadsheet. "
            f"spreadsheet_id={spreadsheet_id}, service_account_email={service_account_email}. "
            "Verify the sheet is shared with this service account as Editor and that "
            "Google Sheets API/Drive API are enabled in the service account project."
        ) from exc

    summary_headers = [
        "run_utc",
        "subreddit",
        "window_label",
        "window_start_utc",
        "window_end_utc",
        "posts_count",
        "comments_count",
        "analysis_provider",
        "analysis_model",
        "average_sentiment",
        "summary_of_week",
        "sentiment_method_notes",
        "top_3_posts_json",
    ]
    keywords_headers = [
        "run_utc",
        "subreddit",
        "window_label",
        "keyword",
        "count",
        "links_json",
    ]

    summary_sheet = _get_or_create_worksheet(spreadsheet, summary_worksheet_name, summary_headers)
    keywords_sheet = _get_or_create_worksheet(spreadsheet, keywords_worksheet_name, keywords_headers)

    run_utc = datetime.now(timezone.utc).isoformat()
    summary_row = [
        run_utc,
        result.get("subreddit", ""),
        result.get("window", {}).get("label", ""),
        result.get("window", {}).get("start_utc", ""),
        result.get("window", {}).get("end_utc", ""),
        result.get("counts", {}).get("posts", 0),
        result.get("counts", {}).get("comments", 0),
        result.get("analysis", {}).get("provider", ""),
        result.get("analysis", {}).get("model", ""),
        result.get("average_sentiment", ""),
        result.get("summary_of_week", ""),
        result.get("sentiment_method_notes", ""),
        json.dumps(result.get("top_3_posts_by_upvotes", []), ensure_ascii=True),
    ]
    summary_sheet.append_row(summary_row, value_input_option="RAW")

    keyword_rows = []
    for keyword, payload in result.get("keyword_mentions", {}).items():
        keyword_rows.append(
            [
                run_utc,
                result.get("subreddit", ""),
                result.get("window", {}).get("label", ""),
                keyword,
                payload.get("count", 0),
                json.dumps(payload.get("links", []), ensure_ascii=True),
            ]
        )

    if keyword_rows:
        keywords_sheet.append_rows(keyword_rows, value_input_option="RAW")

    return {
        "spreadsheet_id": spreadsheet_id,
        "summary_worksheet": summary_worksheet_name,
        "keywords_worksheet": keywords_worksheet_name,
        "run_utc": run_utc,
        "status": "written",
    }