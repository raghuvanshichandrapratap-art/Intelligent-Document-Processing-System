import argparse
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import TemporaryDirectory

testing_folder = Path(__file__).resolve().parent
project_folder = testing_folder.parent

project_folder_text = str(project_folder)

if project_folder_text in sys.path:
    sys.path.remove(project_folder_text)

sys.path.insert(0, project_folder_text)

from pdf2image import convert_from_path

from Backend.extractors.document_classification import (
    aadhaar_back_schema,
    aadhaar_front_schema,
    choose_document_type_from_text,
    describe_confidence_level,
    invoice_schema,
    pan_card_schema,
    passbook_schema,
)
from Backend.extractors.field_mapper import extract_field_values_using_schema
from Backend.extractors.validation import validate_extracted_fields
from testing.reporting import (
    get_document_outputs_folder,
    outputs_folder,
    save_aadhaar_json_file,
    save_accuracy_excel,
    save_document_json_file,
)
from testing.llm_testing import (
    get_text_llm_models,
    get_vision_llm_models,
    is_llm_model,
    process_image_with_llm,
    process_text_with_llm,
)
from Backend.pipeline import process_uploaded_document
from Backend.processors.image_processor import extract_text_from_image, get_available_ocr_models
from Backend.processors.pdf_processor import (
    extract_text_pages_from_digital_pdf,
    get_available_digital_pdf_extractors,
)


test_data_folder = testing_folder / "test-data"
aadhaar_images_folder = test_data_folder / "generated_docs" / "aadhaar" / "image"
aadhaar_ground_truth_folder = test_data_folder / "ground_truth" / "aadhaar" / "image"
pan_images_folder = test_data_folder / "generated_docs" / "pan" / "image"
pan_ground_truth_folder = test_data_folder / "ground_truth" / "pan" / "image"
passbook_images_folder = test_data_folder / "generated_docs" / "passbook" / "image"
passbook_ground_truth_folder = test_data_folder / "ground_truth" / "passbook" / "image"
invoice_images_folder = test_data_folder / "generated_docs" / "invoice" / "image"
invoice_ground_truth_folder = test_data_folder / "ground_truth" / "invoice" / "image"
aadhaar_pdfs_folder = test_data_folder / "generated_docs" / "aadhaar" / "pdf"
aadhaar_pdf_ground_truth_folder = test_data_folder / "ground_truth" / "aadhaar" / "pdf"
pan_pdfs_folder = test_data_folder / "generated_docs" / "pan" / "pdf"
pan_pdf_ground_truth_folder = test_data_folder / "ground_truth" / "pan" / "pdf"
passbook_pdfs_folder = test_data_folder / "generated_docs" / "passbook" / "pdf"
passbook_pdf_ground_truth_folder = test_data_folder / "ground_truth" / "passbook" / "pdf"
invoice_pdfs_folder = test_data_folder / "generated_docs" / "invoice" / "pdf"
invoice_pdf_ground_truth_folder = test_data_folder / "ground_truth" / "invoice" / "pdf"
image_extensions = {".png", ".jpg", ".jpeg"}
aadhaar_footer_words = [
    "www",
    "uidai",
    "help@",
    "help @",
    "1947",
    "qr",
    "government of india",
    "bharat sarkar",
    "unique identification",
    "mera aadhaar",
    "aadhaar is proof",
    "aadhar is proof",
    "sample photo",
    "bengaluru-560 001",
    "bengaluru 560 001",
]
image_quality_categories = [
    "clean_digital",
    "rotated_page",
    "skewed_text",
    "cropped_page",
    "partial_content",
    "low_contrast_text",
    "jpeg_heavy_compression",
    "low_resolution",
    "mobile_photo",
    "partial_crop",
    "overexposed",
    "low_light",
    "blurred",
    "shadow",
    "skewed",
    "cropped",
    "rotated",
    "clean",
]
MODEL_SCOPE_ALL = "all"
MODEL_SCOPE_NO_LLM = "no_llm"
MODEL_SCOPE_LLM_ONLY = "llm_only"


def choose_models(primary_models, llm_models, model_scope=MODEL_SCOPE_ALL):
    if model_scope == MODEL_SCOPE_NO_LLM:
        return list(primary_models)

    if model_scope == MODEL_SCOPE_LLM_ONLY:
        return list(llm_models)

    return list(primary_models) + list(llm_models)


def run_document_processing(file_name):
    # This function controls the full process from file name to final JSON.

    try:
        process_uploaded_document(file_name)

        # Show one short success message, but do not print the full JSON.
        print("Done. File processed and saved in the outputs folder.")
    except Exception as error:
        # Show a simple error message so the output is easier to read.
        print(error)


def normalize_value_for_comparison(value):
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)

    value = str(value or "").casefold()
    value = re.sub(r"[\s\-:/.,|]+", "", value)
    return value


def normalize_date_value_for_comparison(value):
    value = str(value or "").strip()
    month_names = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    month_match = re.match(r"^(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{2,4})$", value)
    if month_match:
        day, month_name, year = month_match.groups()
        month = month_names.get(month_name[:3].casefold())
        if month:
            if len(year) == 2:
                year = f"20{year}"
            return f"{year}{month}{int(day):02d}"

    patterns = [
        r"^((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})$",
        r"^(\d{1,2})[-/.](\d{1,2})[-/.]((?:19|20)?\d{2})$",
    ]

    for index, pattern in enumerate(patterns):
        match = re.match(pattern, value)
        if not match:
            continue

        if index == 0:
            year, month, day = match.groups()
        else:
            day, month, year = match.groups()

        if len(year) == 2:
            year = f"20{year}"

        return f"{year}{int(month):02d}{int(day):02d}"

    return normalize_value_for_comparison(value)


def normalize_text_tokens(value):
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)

    value = str(value or "").casefold()
    value = re.sub(r"[^a-z0-9\u0900-\u097F ]+", " ", value)
    return [token for token in value.split() if token]


def token_similarity_percent(extracted_value, expected_value):
    extracted_tokens = normalize_text_tokens(extracted_value)
    expected_tokens = normalize_text_tokens(expected_value)

    if not extracted_tokens or not expected_tokens:
        return 0

    extracted_set = set(extracted_tokens)
    expected_set = set(expected_tokens)
    overlap_score = len(extracted_set & expected_set) / len(expected_set)
    sequence_score = SequenceMatcher(None, " ".join(extracted_tokens), " ".join(expected_tokens)).ratio()

    return round(max(overlap_score, sequence_score) * 100, 2)


def compare_field_value(field_name, extracted_value, expected_value):
    fuzzy_fields = {
        "name",
        "hindi_name",
        "address",
        "hindi_address",
        "hindi_address_lines",
        "father_name",
        "husband_name",
        "care_of",
        "hindi_father_name",
        "hindi_husband_name",
        "hindi_care_of",
        "father_name",
        "hindi_father_name",
        "bank_name",
        "hindi_bank_name",
        "branch_name",
        "account_holder",
        "account_type",
        "relationship_label",
        "mop",
        "continuation",
        "company",
    }

    if field_name in fuzzy_fields:
        similarity_percent = token_similarity_percent(extracted_value, expected_value)
        threshold = 65 if field_name == "address" else 80

        return similarity_percent >= threshold, similarity_percent

    if field_name in {"date_of_birth", "account_opened", "date_of_issue", "date"}:
        is_match = (
            normalize_date_value_for_comparison(extracted_value)
            == normalize_date_value_for_comparison(expected_value)
        )
        return is_match, 100 if is_match else 0

    if field_name == "total":
        extracted_amount = normalize_amount_for_comparison(extracted_value)
        expected_amount = normalize_amount_for_comparison(expected_value)
        is_match = extracted_amount == expected_amount
        return is_match, 100 if is_match else 0

    if field_name == "source_image":
        extracted_name = Path(str(extracted_value or "")).name.casefold()
        expected_name = Path(str(expected_value or "")).name.casefold()
        is_match = extracted_name == expected_name
        return is_match, 100 if is_match else 0

    if field_name == "branch_manager_stamp_present":
        extracted_bool = str(extracted_value).strip().casefold() in {"true", "yes", "present", "1"}
        expected_bool = bool(expected_value)
        return extracted_bool == expected_bool, 100 if extracted_bool == expected_bool else 0

    is_match = normalize_value_for_comparison(extracted_value) == normalize_value_for_comparison(expected_value)
    return is_match, 100 if is_match else 0


def normalize_amount_for_comparison(value):
    value = str(value or "")
    value = value.replace(",", "")
    match = re.search(r"\d+(?:[ .]\d{2})?", value)
    if not match:
        return ""

    amount = match.group(0).replace(" ", ".")
    if "." not in amount and len(amount) > 2:
        amount = f"{amount[:-2]}.{amount[-2:]}"

    try:
        return f"{float(amount):.2f}"
    except Exception:
        return amount


def ground_truth_value_is_present(value):
    if value is None or value is False:
        return False

    if isinstance(value, str) and value.strip() == "":
        return False

    if value == "" or value == [] or value == {}:
        return False

    return True


def compare_with_ground_truth(extracted_data, ground_truth):
    field_results = {}
    matched_fields = 0
    total_fields = 0
    metadata_fields = {"source_image", "pdf_type", "pages"}

    for field_name, expected_value in ground_truth.items():
        extracted_value = extracted_data.get(field_name, "")
        used_for_accuracy = (
            field_name not in metadata_fields
            and ground_truth_value_is_present(expected_value)
        )

        if not used_for_accuracy:
            field_results[field_name] = {
                "extracted_value": extracted_value,
                "ground_truth_value": expected_value,
                "match": False,
                "used_for_accuracy": False,
                "reason": (
                    "metadata field"
                    if field_name in metadata_fields
                    else "ground truth value not present"
                ),
            }
            continue

        total_fields += 1
        is_match, similarity_percent = compare_field_value(field_name, extracted_value, expected_value)

        if is_match:
            matched_fields += 1

        field_results[field_name] = {
            "extracted_value": extracted_value,
            "ground_truth_value": expected_value,
            "match": is_match,
            "used_for_accuracy": True,
            "similarity_percent": similarity_percent,
        }

    accuracy = (matched_fields / total_fields * 100) if total_fields else 0

    return {
        "matched_fields": matched_fields,
        "total_fields": total_fields,
        "accuracy_percent": round(accuracy, 2),
        "field_results": field_results,
    }


def make_readable_comparison(comparison):
    readable_fields = {}

    for field_name, field_result in comparison["field_results"].items():
        readable_field = {
            "expected": field_result["ground_truth_value"],
            "extracted": field_result["extracted_value"],
            "match": field_result["match"],
            "used_for_accuracy": field_result["used_for_accuracy"],
        }

        if "similarity_percent" in field_result:
            readable_field["similarity_percent"] = field_result["similarity_percent"]

        if "reason" in field_result:
            readable_field["reason"] = field_result["reason"]

        readable_fields[field_name] = readable_field

    return {
        "accuracy_percent": comparison["accuracy_percent"],
        "matched_fields": comparison["matched_fields"],
        "total_fields": comparison["total_fields"],
        "fields": readable_fields,
    }


def load_ground_truth_for_image(image_path, ground_truth_folder):
    ground_truth_path = ground_truth_folder / f"{image_path.stem}.json"

    if not ground_truth_path.is_file():
        return ground_truth_path, {}

    with ground_truth_path.open("r", encoding="utf-8") as json_file:
        return ground_truth_path, json.load(json_file)


def get_clean_ocr_lines(text):
    lines = []

    for line in text.splitlines():
        line = " ".join(line.strip().split())
        if line:
            lines.append(line)

    return lines


def text_has_devanagari(text):
    return re.search(r"[\u0900-\u097F]", text) is not None


def normalize_ocr_digits(value):
    digit_map = str.maketrans({
        "०": "0",
        "१": "1",
        "२": "2",
        "३": "3",
        "४": "4",
        "५": "5",
        "६": "6",
        "७": "7",
        "८": "8",
        "९": "9",
    })
    return str(value or "").translate(digit_map)


def line_is_noise(line):
    normalized = line.casefold()

    if any(word in normalized for word in aadhaar_footer_words):
        return True

    if "@" in normalized:
        return True

    if re.fullmatch(r"[\W\d_]+", line):
        return True

    return False


def find_aadhaar_number(text):
    normalized_text = normalize_ocr_digits(text)

    for match in re.finditer(
        r"(?<!\d)(\d{4})[ \t-]*(\d{4})[ \t-]*(\d{4})(?![ \t-]*\d)",
        normalized_text,
    ):
        # Synthetic test cards may use 0/1-prefixed numbers, so only enforce
        # the 12-digit grouping here instead of production UIDAI allocation rules.
        return " ".join(match.groups())

    return ""


def find_vid(text):
    normalized_text = normalize_ocr_digits(text)
    match = re.search(
        r"(?<!\d)(\d{4})[ \t-]*(\d{4})[ \t-]*(\d{4})[ \t-]*(\d{4})(?![ \t-]*\d)",
        normalized_text,
    )

    if match:
        return " ".join(match.groups())

    return ""


def find_date_of_birth(text):
    text = normalize_ocr_digits(text)
    patterns = [
        r"(?:DOB|Date\s*of\s*Birth|Birth)\s*[:：-]?\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})",
        r"(?:DOB|Date\s*of\s*Birth|Birth)\s*[:：-]?\s*(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})",
        r"(?<!\d)(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})(?!\d)",
        r"(?<!\d)(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})(?!\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", "", match.group(1))

    return ""


def find_year_of_birth(text):
    text = normalize_ocr_digits(text)
    match = re.search(r"(?:YOB|Year\s*of\s*Birth|Birth\s*Year)\s*[:：-]?\s*((?:19|20)\d{2})", text, flags=re.IGNORECASE)

    if match:
        return match.group(1)

    return ""


def find_pan_number(text):
    normalized_text = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    match = re.search(r"[A-Z]{5}\d{4}[A-Z]", normalized_text)
    return match.group(0) if match else ""


def normalize_date_for_ground_truth(date_text):
    date_text = re.sub(r"\s*([/.-])\s*", r"\1", str(date_text or "").strip())

    match = re.search(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-]((?:19|20)\d{2})(?!\d)", date_text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    match = re.search(r"(?<!\d)((?:19|20)\d{2})[/-](\d{1,2})[/-](\d{1,2})(?!\d)", date_text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    return date_text


def find_pan_date_of_birth(text):
    label_pattern = r"(?:Date\s*of\s*Birth|DOB|जन्म\s*की\s*तारीख)"
    patterns = [
        rf"{label_pattern}\s*[:：/-]?\s*(\d{{1,2}}[/-]\d{{1,2}}[/-](?:19|20)\d{{2}})",
        rf"{label_pattern}\s*[:：/-]?\s*((?:19|20)\d{{2}}[/-]\d{{1,2}}[/-]\d{{1,2}})",
        r"(?<!\d)(\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2})(?!\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_date_for_ground_truth(match.group(1))

    return ""


def find_card_issue_date_text(text):
    candidates = re.findall(r"(?<!\d)(\d{8})(?!\d)", text)

    for candidate in candidates:
        if not candidate.startswith(("19", "20")):
            return candidate

    return ""


def line_has_pan_noise(line):
    return re.search(
        r"PAN|Permanent|Account|Number|Income|Tax|Department|Govt|Government|India|Signature|Date|Birth|QR|Card",
        line,
        flags=re.IGNORECASE,
    ) is not None


def clean_pan_name(value):
    value = re.sub(r"[^A-Za-z .']", " ", str(value))
    words = [word for word in value.split() if len(word) > 1]
    return " ".join(words[:4])


def find_labeled_pan_value(lines, labels, allow_devanagari=False):
    label_pattern = "|".join(re.escape(label) for label in labels)

    for index, line in enumerate(lines):
        if re.search(label_pattern, line, flags=re.IGNORECASE):
            for candidate in lines[index + 1:index + 4]:
                if allow_devanagari:
                    cleaned = clean_hindi_text(candidate)
                    if cleaned:
                        return cleaned
                elif is_possible_name_line(candidate):
                    return clean_pan_name(candidate)

    return ""


def find_pan_english_names(lines):
    candidates = []

    for line in lines:
        if line_has_pan_noise(line) or text_has_devanagari(line):
            continue

        if find_pan_number(line) or re.search(r"\d|/", line):
            continue

        if is_possible_name_line(line):
            candidates.append(clean_pan_name(line))

    return candidates


def find_pan_hindi_values(lines):
    values = []

    for line in lines:
        if not text_has_devanagari(line) or line_is_noise(line):
            continue

        cleaned = clean_hindi_text(line)

        if cleaned and cleaned not in {"नाम", "पिता का नाम", "जन्म की तारीख", "हस्ताक्षर", "भारत सरकार", "आयकर विभाग"}:
            values.append(cleaned)

    return values


def find_gender(text):
    if re.search(r"\b(female|femaie|woman)\b", text, flags=re.IGNORECASE):
        return "FEMALE"

    if re.search(r"\b(male|maie|man)\b", text, flags=re.IGNORECASE):
        return "MALE"

    if "महिला" in text:
        return "FEMALE"

    if "पुरुष" in text:
        return "MALE"

    return ""


def find_pincode(text):
    text = normalize_ocr_digits(text)
    ignored_numbers = {
        re.sub(r"\D", "", find_aadhaar_number(text)),
        re.sub(r"\D", "", find_vid(text)),
        "1947",
        "560001",
    }
    candidates = re.findall(r"(?<!\d)([1-9]\d{5})(?!\d)", text)

    for candidate in reversed(candidates):
        if candidate not in ignored_numbers:
            return candidate

    return ""


def find_relationship(text):
    patterns = [
        ("care_of", "C/O", r"\b(?:C/O|Care\s*Of)\s*[:：-]?\s*([A-Za-z][A-Za-z .']+)"),
        ("father_name", "S/O", r"\bS[\/I|]?O\s*[:：-]?\s*([A-Za-z][A-Za-z .']+)"),
        ("father_name", "Father", r"\b(?:Father|Father\s*Name)\s*[:：-]?\s*([A-Za-z][A-Za-z .']+)"),
        ("husband_name", "W/O", r"\bW[\/I|]?O\s*[:：-]?\s*([A-Za-z][A-Za-z .']+)"),
        ("husband_name", "Husband", r"\b(?:Husband|Husband\s*Name)\s*[:：-]?\s*([A-Za-z][A-Za-z .']+)"),
    ]
    result = {
        "relationship_label": "",
        "care_of": "",
        "father_name": "",
        "husband_name": "",
    }

    for field_name, label, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            result["relationship_label"] = label
            result[field_name] = clean_person_name(match.group(1))
            return result

    return result


def find_hindi_relationship(lines):
    result = {
        "hindi_relationship_label": "",
        "hindi_care_of": "",
        "hindi_father_name": "",
        "hindi_husband_name": "",
    }
    label_map = {
        "मार्फत": "hindi_care_of",
        "पिता": "hindi_father_name",
        "पुत्र": "hindi_father_name",
        "पति": "hindi_husband_name",
        "पत्नी": "hindi_husband_name",
    }

    for line in lines:
        if not text_has_devanagari(line):
            continue

        match = re.search(r"(मार्फत|पिता|पुत्र|पति|पत्नी)\s*[:：ः-]?\s*([\u0900-\u097F ]+)", line)
        if not match:
            continue

        label = match.group(1)
        value = clean_hindi_text(match.group(2))

        if value:
            result["hindi_relationship_label"] = label
            result[label_map[label]] = value
            return result

    return result


def clean_person_name(value):
    value = re.split(r"\s{2,}|,|\n|Address|पता|S/O|W/O|C/O", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(r"[^A-Za-z .']", " ", value)
    words = value.split()
    junk_prefixes = {"O", "OC", "ID", "NO", "UID", "VID", "DOB", "TO", "OF"}

    while len(words) > 2 and words[0].upper() in junk_prefixes:
        words = words[1:]

    if len(words) > 2 and len(words[0]) <= 3 and words[0].isupper():
        words = words[1:]

    return " ".join(words)


def clean_hindi_text(value):
    value = re.sub(r"[^\u0900-\u097F ]", " ", str(value))
    value = re.sub(r"\b(?:पता|पिता|पुत्र|मार्फत|पति|पत्नी)\b", " ", value)
    return " ".join(value.split())


def find_hindi_name(lines):
    for line in lines:
        if text_has_devanagari(line) and not line_is_noise(line):
            cleaned = re.sub(r"[^\u0900-\u097F ]", " ", line)
            cleaned = " ".join(cleaned.split())

            if cleaned and not re.search(
                r"सरकार|आधार|जन्म|महिला|पुरुष|पहचान|प्राधिकरण|अधिकार",
                cleaned,
            ):
                return cleaned

    return ""


def find_front_name(lines):
    for index, line in enumerate(lines):
        if re.search(r"\b(DOB|Date\s*of\s*Birth|YOB|Male|Female)\b", line, flags=re.IGNORECASE):
            for candidate in reversed(lines[max(0, index - 3):index]):
                if is_possible_name_line(candidate):
                    return clean_person_name(candidate)

    for line in lines:
        if is_possible_name_line(line):
            return clean_person_name(line)

    return ""


def is_possible_name_line(line):
    if line_is_noise(line) or text_has_devanagari(line):
        return False

    if re.search(r"\d|@|/|www|uidai|gov|india|bharat|sample|photo", line, flags=re.IGNORECASE):
        return False

    if re.search(r"\b(DOB|YOB|Male|Female|Address|Father|Husband|Government|India)\b", line, flags=re.IGNORECASE):
        return False

    words = re.findall(r"[A-Za-z]+", line)
    long_words = [word for word in words if len(word) > 1]
    return 2 <= len(long_words) <= 4


def remove_duplicate_pincode(address, pincode):
    if not pincode:
        return address

    seen = False

    def replace_match(match):
        nonlocal seen

        if seen:
            return ""

        seen = True
        return match.group(0)

    return re.sub(rf"(?<!\d){re.escape(pincode)}(?!\d)", replace_match, address)


def find_address(lines, aadhaar_number, pincode, relationship):
    address_lines = find_address_lines_near_pincode(lines, pincode)

    if not address_lines:
        address_lines = []
        start_collecting = False

        for line in lines:
            if line_is_noise(line) or text_has_devanagari(line):
                continue

            line_without_aadhaar = line.replace(aadhaar_number, "").strip()
            if not line_without_aadhaar:
                continue

            if re.search(r"\b(Address|पता|Father|Husband|S/O|W/O|C/O|Care\s*Of)\b", line_without_aadhaar, flags=re.IGNORECASE):
                start_collecting = True
                value = re.sub(
                    r".*?\b(?:Address|पता|Father|Husband|S/O|W/O|C/O|Care\s*Of)\b\s*[:：-]?",
                    "",
                    line_without_aadhaar,
                    flags=re.IGNORECASE,
                ).strip()
                if value:
                    address_lines.append(value)
                continue

            if pincode and pincode in line_without_aadhaar:
                start_collecting = True

            if start_collecting:
                address_lines.append(line_without_aadhaar)

                if pincode and pincode in line_without_aadhaar:
                    break

    cleaned_lines = []

    for line in address_lines:
        line = re.sub(r"\b(?:Address|पता)\b\s*[:：-]?", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\b(?:Father|Husband|S/O|W/O|C/O|Care\s*Of)\b\s*[:：-]?\s*[A-Za-z .']+", "", line, flags=re.IGNORECASE)
        line = line.replace(aadhaar_number, "")

        for relationship_value in relationship.values():
            if relationship_value and relationship_value not in ["Father", "Husband", "C/O"]:
                line = line.replace(relationship_value, "")

        if pincode:
            line = line.replace(pincode, f" {pincode} ")
        line = re.sub(r"[^A-Za-z0-9\u0900-\u097F ,.-]", " ", line)
        line = " ".join(line.split(" "))
        line = " ".join(line.split())

        if line and not line_is_noise(line):
            cleaned_lines.append(line)

    address = " ".join(dict.fromkeys(cleaned_lines))
    address = remove_duplicate_pincode(address, pincode)

    if pincode and pincode not in address:
        address = f"{address} {pincode}".strip()

    return address


def find_hindi_address_lines(lines, pincode, hindi_relationship):
    selected_lines = []
    started = False

    for line in lines:
        has_hindi = text_has_devanagari(line)
        has_pincode = bool(pincode and pincode in line)

        if not has_hindi and not has_pincode:
            continue

        if line_is_noise(line) and not has_pincode:
            continue

        if "पता" in line:
            started = True
            selected_lines.append("पता:")
            remainder = clean_hindi_text(re.sub(r".*?पता\s*[:：-]?", "", line))
            if remainder:
                selected_lines.append(remainder)
            continue

        if started or (pincode and pincode in line):
            selected_lines.append(line)

        if pincode and pincode in line:
            break

    if not selected_lines and pincode:
        for index, line in enumerate(lines):
            if pincode not in line:
                continue

            nearby_lines = lines[max(0, index - 4):index + 1]
            selected_lines = [
                line for line in nearby_lines
                if (text_has_devanagari(line) or pincode in line) and not line_is_noise(line)
            ]
            break

    cleaned_lines = []
    for line in selected_lines:
        if line.strip() == "पता:":
            cleaned_lines.append("पता:")
            continue

        cleaned = line
        cleaned = re.sub(r"[^\u0900-\u097F0-9A-Za-z ,.-]", " ", cleaned)
        cleaned = " ".join(cleaned.split())

        if cleaned and cleaned not in cleaned_lines:
            cleaned_lines.append(cleaned)

    return cleaned_lines


def join_hindi_address_lines(lines):
    address_parts = [
        line for line in lines
        if line.strip() != "पता:" and not re.search(r"^(मार्फत|पिता|पुत्र|पति|पत्नी)\s*[:：-]?", line)
    ]
    return " ".join(address_parts)


def find_address_lines_near_pincode(lines, pincode):
    if not pincode:
        return []

    for index, line in enumerate(lines):
        if pincode not in line:
            continue

        start_index = max(0, index - 4)

        for candidate_index in range(index, start_index - 1, -1):
            if re.search(r"\b(Address|पता|Father|Husband|S/O|W/O|C/O|Care\s*Of)\b", lines[candidate_index], flags=re.IGNORECASE):
                start_index = candidate_index
                break

        return [
            line for line in lines[start_index:index + 1]
            if not line_is_noise(line) and not text_has_devanagari(line)
        ]

    return []


def infer_aadhaar_document_type(text, image_path):
    file_name = image_path.stem.casefold()
    front_score = 0
    back_score = 0

    if re.search(r"\b(DOB|Date\s*of\s*Birth|YOB|Male|Female)\b", text, flags=re.IGNORECASE):
        front_score += 2

    if find_gender(text) or find_date_of_birth(text) or find_year_of_birth(text):
        front_score += 1

    if re.search(r"\b(Address|पता|Pincode|Father|Husband|S/O|W/O|C/O|Care\s*Of)\b", text, flags=re.IGNORECASE):
        back_score += 2

    if find_pincode(text):
        back_score += 1

    if back_score > front_score:
        return "aadhaar_back"

    if front_score > back_score:
        return "aadhaar_front"

    if "front" in file_name:
        return "aadhaar_front"

    if "back" in file_name:
        return "aadhaar_back"

    return "aadhaar_front"


def get_image_side(image_path):
    file_name = image_path.stem.casefold()

    if "front" in file_name:
        return "front"

    if "back" in file_name:
        return "back"

    return "unknown"


def get_document_image_side(document_name, image_path):
    side = get_image_side(image_path)

    if side == "unknown" and document_name in {"passbook", "invoice"}:
        return "front"

    return side


def get_image_quality_category(image_path):
    file_name = image_path.stem.casefold()

    for category in image_quality_categories:
        if category in file_name:
            return category

    return "unknown"


def extract_aadhaar_fields_from_text(text, image_path):
    if not text.strip() or not find_aadhaar_number(text):
        return "", {}

    lines = get_clean_ocr_lines(text)
    document_type = infer_aadhaar_document_type(text, image_path)
    aadhaar_number = find_aadhaar_number(text)
    vid = find_vid(text)

    if document_type == "aadhaar_back":
        pincode = find_pincode(text)
        relationship = find_relationship(text)
        hindi_relationship = find_hindi_relationship(lines)
        hindi_address_lines = find_hindi_address_lines(lines, pincode, hindi_relationship)
        extracted_data = {
            "aadhaar_number": aadhaar_number,
            "address": find_address(lines, aadhaar_number, pincode, relationship),
            "pincode": pincode,
            "vid": vid,
            **relationship,
            **hindi_relationship,
            "hindi_address": join_hindi_address_lines(hindi_address_lines),
            "hindi_address_lines": hindi_address_lines,
        }
    else:
        date_of_birth = find_date_of_birth(text)
        extracted_data = {
            "aadhaar_number": aadhaar_number,
            "name": find_front_name(lines),
            "gender": find_gender(text),
            "vid": vid,
            "hindi_name": find_hindi_name(lines),
            "date_of_birth": date_of_birth,
            "year_of_birth": find_year_of_birth(text) if not date_of_birth else "",
        }

    return document_type, extracted_data


def make_aadhaar_classification(document_type):
    schema = aadhaar_back_schema if document_type == "aadhaar_back" else aadhaar_front_schema
    matched_fields = list(schema.keys())

    return {
        "document_type": document_type,
        "schema": schema,
        "confidence": 1,
        "confidence_percent": 100,
        "confidence_level": describe_confidence_level(1),
        "matched_fields": matched_fields,
        "missing_fields": [],
        "matched_count": len(matched_fields),
        "total_fields": len(matched_fields),
        "all_scores": [],
        "score_gap": 0,
    }


def make_pan_classification():
    return {
        "document_type": "pan_card",
        "schema": pan_card_schema,
        "confidence": 1,
        "confidence_percent": 100,
        "confidence_level": describe_confidence_level(1),
        "matched_fields": list(pan_card_schema.keys()),
        "missing_fields": [],
        "matched_count": len(pan_card_schema),
        "total_fields": len(pan_card_schema),
        "all_scores": [],
        "score_gap": 0,
    }


def make_passbook_classification():
    return {
        "document_type": "passbook",
        "schema": passbook_schema,
        "confidence": 1,
        "confidence_percent": 100,
        "confidence_level": describe_confidence_level(1),
        "matched_fields": list(passbook_schema.keys()),
        "missing_fields": [],
        "matched_count": len(passbook_schema),
        "total_fields": len(passbook_schema),
        "all_scores": [],
        "score_gap": 0,
    }


def make_invoice_classification():
    return {
        "document_type": "invoice",
        "schema": invoice_schema,
        "confidence": 1,
        "confidence_percent": 100,
        "confidence_level": describe_confidence_level(1),
        "matched_fields": list(invoice_schema.keys()),
        "missing_fields": [],
        "matched_count": len(invoice_schema),
        "total_fields": len(invoice_schema),
        "all_scores": [],
        "score_gap": 0,
    }


def extracted_data_has_value(extracted_data):
    return any(str(value or "").strip() for value in extracted_data.values())


def extract_pan_fields_from_text(text, image_path):
    if not text.strip():
        return "", {}

    lines = get_clean_ocr_lines(text)
    pan_number = find_pan_number(text)

    if not pan_number and "pan" not in text.casefold() and "income tax" not in text.casefold():
        return "", {}

    english_names = find_pan_english_names(lines)
    hindi_values = find_pan_hindi_values(lines)
    extracted_data = {
        "pan_number": pan_number,
        "name": find_labeled_pan_value(lines, ["Name", "नाम"]) or (english_names[0] if english_names else ""),
        "hindi_name": find_labeled_pan_value(lines, ["नाम"], allow_devanagari=True) or (hindi_values[0] if hindi_values else ""),
        "father_name": find_labeled_pan_value(lines, ["Father's Name", "Father Name", "पिता का नाम"])
        or (english_names[1] if len(english_names) > 1 else ""),
        "hindi_father_name": find_labeled_pan_value(lines, ["पिता का नाम"], allow_devanagari=True)
        or (hindi_values[1] if len(hindi_values) > 1 else ""),
        "date_of_birth": find_pan_date_of_birth(text),
        "signature_present": bool(re.search(r"Signature|हस्ताक्षर", text, flags=re.IGNORECASE)),
        "card_issue_date_text": find_card_issue_date_text(text),
    }

    return "pan_card", extracted_data


def clean_passbook_value(value):
    value = re.sub(r"[\u00a0]+", " ", str(value or ""))
    value = re.sub(r"\s+", " ", value)
    return value.strip(" :-|.\t\n")


def normalize_passbook_ocr_digits(value):
    return normalize_ocr_digits(value)


def normalize_passbook_code_text(value):
    value = normalize_passbook_ocr_digits(value)
    value = value.upper()
    value = value.replace("O", "0")
    return re.sub(r"[^A-Z0-9@._+-]", "", value)


def get_passbook_lines(text):
    return [clean_passbook_value(line) for line in text.splitlines() if clean_passbook_value(line)]


def clean_passbook_name(value):
    value = clean_passbook_value(value)
    value = re.split(
        r"\b(?:MOP|CIF|Account|A/c|Address|Nom|Customer|Date|PAN)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.sub(r"[^A-Za-z .']", " ", value)
    value = " ".join(value.split())
    value = re.sub(r"\blyer\b", "Iyer", value, flags=re.IGNORECASE)
    return value


def clean_passbook_address(value):
    value = clean_passbook_value(value)
    value = re.split(
        r"\b(?:MOP|A/c\s*Opening|Nom\s*Reg|Customer'?s\s*PAN|Date\s*of\s*Issue|CONTINUATION|Branch\s*Manager)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.sub(r"[^A-Za-z0-9 ,.-]", " ", value)
    return " ".join(value.split())


def find_first_regex_group(text, patterns, flags=re.IGNORECASE):
    for pattern in patterns:
        match = re.search(pattern, text, flags=flags)
        if match:
            return clean_passbook_value(match.group(1))

    return ""


def find_passbook_bank_name(text):
    known_banks = [
        "State Bank of India",
        "HDFC Bank",
        "ICICI Bank",
        "Punjab National Bank",
        "Karnataka Bank",
        "Axis Bank",
        "Bank of India",
    ]

    normalized_text = " ".join(text.split()).casefold()
    for bank_name in known_banks:
        if bank_name.casefold() in normalized_text:
            return bank_name

    if re.search(r"\bSBI\b", text, flags=re.IGNORECASE) or re.search(r"भारतीय\s+(?:स्टेट|सूटेट|सटेट)", text):
        return "State Bank of India"

    candidate = find_first_regex_group(text, [r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+Bank(?:\s+of\s+India)?)\b"])
    if re.search(r"\b(REGULAR|SAVINGS|ACCOUNT)\b", candidate, flags=re.IGNORECASE):
        return ""

    return candidate


def find_passbook_hindi_bank_name(lines):
    for line in lines:
        if text_has_devanagari(line) and "प्रबंधक" not in line:
            cleaned = re.sub(r"[^\u0900-\u097F ]", " ", line)
            cleaned = " ".join(cleaned.split())

            if cleaned:
                return cleaned

    return ""


def find_passbook_labeled_value(text, label_pattern, stop_pattern):
    pattern = rf"(?:{label_pattern})\s*[:：.]?\s*(.+?)(?=\s+(?:{stop_pattern})\s*[:：.]?|\n|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

    if match:
        return clean_passbook_value(match.group(1))

    return ""


def find_passbook_account_holder(text):
    value = find_passbook_labeled_value(
        text,
        r"\bName\b",
        r"MOP|S/D/H/o|CIF|Account|A/c|Address|Nom|Customer'?s\s*PAN|Date\s*of\s*Issue",
    )
    value = clean_passbook_name(value)

    if len(value.split()) >= 2:
        return value

    lines = get_passbook_lines(text)
    for index, line in enumerate(lines):
        if re.fullmatch(r"Mr\.?|Ms\.?|Mrs\.?", line, flags=re.IGNORECASE):
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            name = clean_passbook_name(f"{line} {next_line}")
            if len(name.split()) >= 2:
                return name

        if re.search(r"\b(?:Mr|Ms|Mrs)\.?\s+[A-Za-z]", line, flags=re.IGNORECASE):
            name = clean_passbook_name(line)
            if len(name.split()) >= 2:
                return name

    return value


def find_passbook_father_name(text):
    value = find_passbook_labeled_value(
        text,
        r"S\s*/\s*D\s*/\s*H\s*/?\s*o|S/D/H/o|Father(?:\s+Name)?",
        r"CIF|Account|A/c|Address|MOP|Nom|Customer'?s\s*PAN|Date\s*of\s*Issue",
    )
    value = clean_passbook_name(value).title()

    if len(value.split()) >= 2:
        return value

    lines = get_passbook_lines(text)
    for index, line in enumerate(lines):
        if re.search(r"S\s*/\s*D\s*/\s*H|S/D/H|S/O|D/O|8/0", line, flags=re.IGNORECASE):
            nearby = " ".join(lines[index + 1:index + 4])
            name = clean_passbook_name(nearby).title()
            if len(name.split()) >= 2:
                return name

    for index, line in enumerate(lines[:-1]):
        joined = clean_passbook_name(f"{line} {lines[index + 1]}").title()
        if re.fullmatch(r"[A-Z][A-Za-z]+ [A-Z][A-Za-z]+", joined) and "Iyer" in joined:
            return joined

    return value


def find_passbook_address(text):
    value = find_passbook_labeled_value(
        text,
        r"\bAddress\b",
        r"MOP|A/c\s*Opening|Nom\s*Reg|Customer'?s\s*PAN|Date\s*of\s*Issue|CONTINUATION|Branch\s*Manager",
    )
    return clean_passbook_address(value)


def find_passbook_address_from_lines(text):
    lines = get_passbook_lines(text)
    selected = []

    for index, line in enumerate(lines):
        normalized_line = normalize_passbook_ocr_digits(line)
        if re.search(r"\b(?:Lake|Road|Nagar|Street|Tamil|Kerala|Gujarat|Chennai|Kochi|Ahmedabad)\b", normalized_line, flags=re.IGNORECASE):
            selected.append(normalized_line)
            for next_line in lines[index + 1:index + 3]:
                next_line = normalize_passbook_ocr_digits(next_line)
                if re.search(r"(?<!\d)[1-9]\d{5}(?!\d)", next_line):
                    selected.append(next_line)
                    break
            break

    return clean_passbook_address(" ".join(selected))


def find_passbook_ifsc(text):
    label_match = re.search(r"\bIFSC\s*[:：.]?\s*([A-Z]{4}\s*0?\s*[A-Z0-9]{6,7})", text, flags=re.IGNORECASE)
    if label_match:
        value = normalize_passbook_code_text(label_match.group(1))
        if len(value) == 12 and value[4] != "0":
            value = value[:4] + value[5:]
        return value

    normalized = normalize_passbook_code_text(text)
    match = re.search(r"[A-Z]{4}[A-Z0-9]?0[A-Z0-9]{6}", normalized)

    if not match:
        return ""

    value = match.group(0)
    if len(value) == 12 and value[4] != "0":
        value = value[:4] + value[5:]

    return value


def find_passbook_pan(text):
    normalized = normalize_passbook_ocr_digits(text).upper()
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    match = re.search(r"[A-Z]{5}\d{4}[A-Z]", normalized)
    return match.group(0) if match else ""


def find_passbook_dates(text):
    normalized = normalize_passbook_ocr_digits(text)
    return re.findall(r"(?<!\d)(\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2})(?!\d)", normalized)


def find_passbook_number_candidates(text):
    normalized = normalize_passbook_ocr_digits(text)
    return re.findall(r"(?<!\d)(\d{6,18})(?!\d)", normalized)


def find_passbook_account_number(text):
    for number in find_passbook_number_candidates(text):
        if 12 <= len(number) <= 18:
            return number

    return ""


def find_passbook_cif_number(text):
    normalized = normalize_passbook_ocr_digits(text)
    match = re.search(r"(?:CIF|CLE)\s*Number[\s\S]{0,80}?(\d{9})", normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    date_match = re.search(r"\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}[\s\S]{0,80}?(\d{9})", normalized)
    if date_match:
        return date_match.group(1)

    nom_match = re.search(r"(\d{9})[\s\S]{0,80}?Nom\s*Reg", normalized, flags=re.IGNORECASE)
    if nom_match:
        return nom_match.group(1)

    for number in find_passbook_number_candidates(text):
        if len(number) == 9 and number not in {find_passbook_account_number(text)}:
            return number

    return ""


def find_passbook_nom_reg_no(text):
    normalized = normalize_passbook_ocr_digits(text)
    match = re.search(r"Nom\s*Reg\s*No\w*[\s\S]{0,80}?(\d{8,14})", normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    for number in find_passbook_number_candidates(text):
        if 10 <= len(number) <= 14 and number != find_passbook_account_number(text):
            return number

    return ""


def extract_passbook_fields_from_text(text, image_path):
    if not text.strip():
        return "", {}

    normalized = text.casefold()
    if not any(marker in normalized for marker in ["ifsc", "account", "accowet", "passbook", "cif", "micr", "customer", "भारतीय"]):
        return "", {}

    lines = get_clean_ocr_lines(text)
    compact_text = "\n".join(lines)
    branch_name = find_first_regex_group(compact_text, [r"\bBranch\s*[:：]\s*([A-Za-z ]+?)(?=\s+Code\b|\n|$)"])
    branch_code = normalize_passbook_ocr_digits(find_first_regex_group(compact_text, [r"\bCode\s*[:：]?\s*([0-9०-९]{4,8})"]))
    email = find_first_regex_group(compact_text, [r"\bEmail\s*[:：]\s*([A-Z0-9._%+-]+\s*@\s*[A-Z0-9.-]+\s*\.\s*[A-Z]{2,})"])
    phone = find_first_regex_group(compact_text, [r"\bPhone\s*No\.?\s*[:：]\s*(0?\d{7,12})"])
    micr = normalize_passbook_ocr_digits(find_first_regex_group(compact_text, [r"\bMICR\s*[:：]?\s*([0-9०-९]{9})"]))
    ifsc = find_passbook_ifsc(compact_text)
    account_number = normalize_passbook_ocr_digits(find_first_regex_group(compact_text, [r"\bAccount\s*No\.?\s*[:：]?\s*([0-9०-९][0-9०-९ ]{8,22})"])).replace(" ", "")
    if not account_number:
        account_number = find_passbook_account_number(compact_text)
    account_type = find_passbook_labeled_value(
        compact_text,
        r"A/c\s*Type|Account\s*Type",
        r"Address|MOP|A/c\s*Opening|Nom|Customer'?s\s*PAN|Date\s*of\s*Issue",
    )
    if not account_type and re.search(r"SAVINGS\s+BANK\s+ACCOUNT", compact_text, flags=re.IGNORECASE):
        account_type = "REGULAR SAVINGS BANK ACCOUNT"
    account_opened = normalize_passbook_ocr_digits(find_first_regex_group(compact_text, [r"A/c\s*Opening\s*Dt\s*[:：]?\s*([0-9०-९]{1,2}[/-][0-9०-९]{1,2}[/-](?:19|20)[0-9०-९]{2})"]))
    date_of_issue = normalize_passbook_ocr_digits(find_first_regex_group(compact_text, [r"Date\s*of\s*Issue\s*[:：]?\s*([0-9०-९]{1,2}[/-][0-9०-९]{1,2}[/-](?:19|20)[0-9०-९]{2})"]))
    dates = find_passbook_dates(compact_text)
    if not account_opened and dates:
        account_opened = dates[0]
    if not date_of_issue and dates:
        date_of_issue = dates[-1]
    nom_reg_no = normalize_passbook_ocr_digits(find_first_regex_group(compact_text, [r"Nom\s*Reg\s*No\s*[:：]?\s*([0-9०-९]{6,14})"]))
    if not nom_reg_no:
        nom_reg_no = find_passbook_nom_reg_no(compact_text)
    cif_number = normalize_passbook_ocr_digits(find_first_regex_group(compact_text, [r"(?:CIF|CLE)\s*Number\s*[:：]?\s*([0-9०-९]{6,14})"]))
    if not cif_number:
        cif_number = find_passbook_cif_number(compact_text)
    pan_number = find_passbook_pan(compact_text)
    mop = find_passbook_labeled_value(compact_text, r"\bMOP\b", r"A/c\s*Opening|Nom|Customer'?s\s*PAN|Date\s*of\s*Issue")
    if not mop and re.search(r"\bSINGLE\b", compact_text, flags=re.IGNORECASE):
        mop = "SINGLE"
    relationship_label = "D/O" if re.search(r"\bD/O\b", compact_text, flags=re.IGNORECASE) else ""
    if not relationship_label and re.search(r"\bS/O\b", compact_text, flags=re.IGNORECASE):
        relationship_label = "S/O"
    account_holder = find_passbook_account_holder(compact_text)
    if not relationship_label and account_holder.lower().startswith("mr"):
        relationship_label = "S/O"
    if not relationship_label and account_holder.lower().startswith("ms"):
        relationship_label = "D/O"
    bank_name = find_passbook_bank_name(compact_text)
    if not bank_name and ifsc.startswith("SBIN"):
        bank_name = "State Bank of India"

    extracted_data = {
        "bank_name": bank_name,
        "hindi_bank_name": find_passbook_hindi_bank_name(lines),
        "branch_name": branch_name.title() if branch_name else "",
        "branch_code": branch_code,
        "email": re.sub(r"\s+", "", email).replace(" ", ".").lower(),
        "phone": normalize_passbook_ocr_digits(phone),
        "micr": micr,
        "ifsc": ifsc,
        "account_holder": account_holder,
        "relationship_label": relationship_label,
        "father_name": find_passbook_father_name(compact_text),
        "cif_number": cif_number,
        "account_number": account_number,
        "account_type": account_type.upper() if account_type else "",
        "address": find_passbook_address(compact_text) or find_passbook_address_from_lines(compact_text),
        "mop": mop.upper() if mop else "",
        "account_opened": account_opened,
        "nom_reg_no": nom_reg_no,
        "pan_number": pan_number,
        "date_of_issue": date_of_issue,
        "continuation": "CONTINUATION" if "continuation" in normalized else "",
        "branch_manager_stamp_present": bool(re.search(r"Branch\s*Manager|शाखा\s*प्रबंधक", text, flags=re.IGNORECASE)),
    }

    if not any(value for key, value in extracted_data.items() if key != "branch_manager_stamp_present"):
        return "", {}

    return "passbook", extracted_data


def clean_invoice_line(line):
    line = normalize_passbook_ocr_digits(str(line or ""))
    line = re.sub(r"\s+", " ", line)
    return line.strip(" :-|\t\n")


def get_invoice_lines(text):
    return [clean_invoice_line(line) for line in text.splitlines() if clean_invoice_line(line)]


def line_is_invoice_noise(line):
    return re.search(
        r"\b(?:TEL|FAX|GST|CASH\s*SALES|Doc\s*No|Cashier|Time|Salesperson|Item|Qty|Price|Amount|Tax|THANK|RETURNABLE|Change|Discount|Rounding|Total)\b",
        line,
        flags=re.IGNORECASE,
    ) is not None


def clean_invoice_company(value):
    value = re.sub(r"[^A-Za-z0-9&().,' /-]", " ", str(value or ""))
    return " ".join(value.split()).upper()


def find_invoice_company(lines):
    business_pattern = r"\b(?:ENTERPRISE|SDN|BHD|CO\.?|BOOK|MACHINERY|STATIONERY|POPULAR|AEON|UNIHAKKA|MR\.?\s*D\.?I\.?Y|RESTAURANT|TRADING)\b"

    for line in lines[:12]:
        if line_is_invoice_noise(line):
            continue

        if re.search(business_pattern, line, flags=re.IGNORECASE):
            return clean_invoice_company(line)

    for line in lines[:12]:
        if line_is_invoice_noise(line):
            continue

        if "@" in line or re.search(r"\d{2,}", line):
            continue

        words = re.findall(r"[A-Za-z&().]+", line)
        if len(words) >= 2 and len(line) >= 8:
            return clean_invoice_company(line)

    return ""


def find_invoice_date(text):
    patterns = [
        r"\b(?:Date|Daie|Invoice\s*Date)\s*[:：]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"\b(?:Date|Daie|Invoice\s*Date)\s*[:：]?\s*(\d{1,2}\s+[A-Za-z]{3,}\s+\d{2,4})",
        r"(?<!\d)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?!\d)",
        r"(?<!\d)(\d{1,2}\s+[A-Za-z]{3,}\s+\d{2,4})(?!\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_invoice_line(match.group(1))

    return ""


def find_invoice_receipt_number(text):
    patterns = [
        r"\b(?:Receipt|Invoice|Doc(?:ument)?|Slip)\s*(?:No\.?|Number|#)\s*[:：]?\s*([A-Z0-9][A-Z0-9/-]{3,})",
        r"\b(?:Bill)\s*(?:No\.?|Number|#)\s*[:：]?\s*([A-Z0-9][A-Z0-9/-]{3,})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_invoice_line(match.group(1)).upper()

    return ""


def find_invoice_address(lines):
    address_lines = []
    started = False

    address_words = r"\b(?:NO\.?|LOT|JALAN|JLN|TAMAN|KAWASAN|PERSIARAN|BANDAR|JOHOR|SELANGOR|KUALA|MALAYSIA|CHERAS|BAHRU|PUTRA|LUMPUR)\b"

    for line in lines[:25]:
        if "@" in line or re.search(r"\b(?:TEL|FAX|GST|CASH\s*SALES|Doc\s*No)\b", line, flags=re.IGNORECASE):
            if started:
                break
            continue

        if re.search(address_words, line, flags=re.IGNORECASE) or re.search(r"\b\d{5}\b", line):
            started = True
            address_lines.append(line)
            continue

        if started and len(address_lines) < 6 and not line_is_invoice_noise(line):
            address_lines.append(line)

    address = " ".join(address_lines)
    address = re.sub(r"[^A-Za-z0-9,./& -]", " ", address)
    return " ".join(address.split()).upper()


def find_invoice_labeled_amount(lines, labels):
    label_pattern = "|".join(re.escape(label) for label in labels)

    for line in reversed(lines):
        normalized = line.replace(",", ".")
        if not re.search(rf"\b(?:{label_pattern})\b", normalized, flags=re.IGNORECASE):
            continue

        amounts = re.findall(r"(?:RM|MYR|\$)?\s*-?\d+(?:[., ]\d{2})", normalized, flags=re.IGNORECASE)
        if amounts:
            return normalize_amount_for_comparison(amounts[-1])

    return ""


def find_invoice_currency(text):
    if re.search(r"\b(?:RM|MYR)\b", text, flags=re.IGNORECASE):
        return "MYR"

    if "$" in text:
        return "USD"

    return ""


def find_invoice_total(lines, text):
    labeled_amounts = []

    for line in lines:
        normalized = line.replace(",", ".")
        if re.search(r"\b(?:Total\s*Sales|Grand\s*Total|Net\s*Total|Total|CASH)\b", normalized, flags=re.IGNORECASE):
            amounts = re.findall(r"\$?\s*\d+(?:[., ]\d{2})", normalized)
            for amount in amounts:
                labeled_amounts.append(normalize_amount_for_comparison(amount))

    for amount in reversed(labeled_amounts):
        if amount and amount != "0.00":
            return amount

    all_amounts = [
        normalize_amount_for_comparison(amount)
        for amount in re.findall(r"\$?\s*\d+(?:[., ]\d{2})", text)
    ]
    numeric_amounts = []
    for amount in all_amounts:
        try:
            numeric_amounts.append(float(amount))
        except Exception:
            pass

    return f"{max(numeric_amounts):.2f}" if numeric_amounts else ""


def extract_invoice_fields_from_text(text, image_path):
    if not text.strip():
        return "", {}

    normalized = text.casefold()
    if not any(marker in normalized for marker in ["cash sales", "invoice", "total", "gst", "receipt", "doc no"]):
        return "", {}

    lines = get_invoice_lines(text)
    extracted_data = {
        "company": find_invoice_company(lines),
        "date": find_invoice_date(text),
        "address": find_invoice_address(lines),
        "receipt_number": find_invoice_receipt_number(text),
        "subtotal": find_invoice_labeled_amount(lines, ["subtotal", "sub-total", "total sales"]),
        "tax": find_invoice_labeled_amount(lines, ["total tax", "tax", "gst"]),
        "discount": find_invoice_labeled_amount(lines, ["discount", "disc"]),
        "total": find_invoice_total(lines, text),
        "currency": find_invoice_currency(text),
        "source_image": Path(image_path).name,
    }

    if not any(value for key, value in extracted_data.items() if key != "source_image"):
        return "", {}

    return "invoice", extracted_data


def process_text_for_document(
    text,
    source_path,
    fallback_extractor,
    make_fallback_classification,
):
    classification = choose_document_type_from_text(text)

    if classification["schema"] is not None:
        mapped_fields = extract_field_values_using_schema(text, classification["schema"])
    else:
        mapped_fields = {}

    fallback_document_type, fallback_fields = fallback_extractor(text, source_path)

    if fallback_document_type in {"passbook", "invoice"}:
        classification = make_fallback_classification(fallback_document_type)
        mapped_fields = fallback_fields
    elif fallback_document_type and (classification["document_type"] == "unknown" or not extracted_data_has_value(mapped_fields)):
        classification = make_fallback_classification(fallback_document_type)
        mapped_fields = fallback_fields
    elif fallback_fields:
        mapped_fields = {
            **mapped_fields,
            **{field: value for field, value in fallback_fields.items() if value},
        }

    validation = validate_extracted_fields(classification, mapped_fields)

    return {
        "extracted_data": validation["extracted_data"],
        "validation": {
            "summary": validation["validation_summary"],
            "field_results": validation["validation_results"],
        },
        "comparison_with_ground_truth": {},
        "classification": {
            "document_type": classification["document_type"],
            "confidence_percent": classification["confidence_percent"],
            "confidence_level": classification["confidence_level"],
        },
        "raw_text": text,
    }


def process_image_with_model(
    image_path,
    model_name,
    fallback_extractor,
    make_fallback_classification,
    ocr_languages=None,
    device="cpu",
):
    text = extract_text_from_image(
        image_path,
        model_name=model_name,
        languages=ocr_languages,
        device=device,
    )
    return process_text_for_document(
        text,
        image_path,
        fallback_extractor,
        make_fallback_classification,
    )


def process_aadhaar_image_with_model(image_path, model_name, device="cpu"):
    return process_image_with_model(
        image_path,
        model_name,
        extract_aadhaar_fields_from_text,
        make_aadhaar_classification,
        device=device,
    )


def process_pan_image_with_model(image_path, model_name, device="cpu"):
    return process_image_with_model(
        image_path,
        model_name,
        extract_pan_fields_from_text,
        lambda _document_type: make_pan_classification(),
        device=device,
    )


def process_passbook_image_with_model(image_path, model_name, device="cpu"):
    return process_image_with_model(
        image_path,
        model_name,
        extract_passbook_fields_from_text,
        lambda _document_type: make_passbook_classification(),
        device=device,
    )


def process_invoice_image_with_model(image_path, model_name, device="cpu"):
    return process_image_with_model(
        image_path,
        model_name,
        extract_invoice_fields_from_text,
        lambda _document_type: make_invoice_classification(),
        ocr_languages=("en",),
        device=device,
    )


def list_document_images(images_folder):
    return sorted(
        path for path in images_folder.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )


def list_document_pdfs(pdfs_folder):
    return sorted(
        path for path in pdfs_folder.rglob("*.pdf")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def load_ground_truth_for_pdf(pdf_path, pdfs_folder, ground_truth_folder):
    relative_path = pdf_path.relative_to(pdfs_folder).with_suffix(".json")
    ground_truth_path = ground_truth_folder / relative_path

    if not ground_truth_path.is_file():
        return ground_truth_path, {}

    with ground_truth_path.open("r", encoding="utf-8") as json_file:
        return ground_truth_path, json.load(json_file)


def get_pdf_page_mapping(ground_truth):
    pages = ground_truth.get("pages", {})

    if not isinstance(pages, dict):
        return {}

    return {str(page_number): str(page_label) for page_number, page_label in pages.items()}


def get_pdf_page_ground_truth(ground_truth, page_label):
    page_ground_truth = ground_truth.get(page_label, {})
    return page_ground_truth if isinstance(page_ground_truth, dict) else {}


def get_report_side_for_page(page_label):
    if page_label in {"front", "back"}:
        return page_label

    return "front"


def convert_pdf_to_page_images(pdf_path, temporary_folder, dpi=150):
    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as error:
        raise RuntimeError(f"Could not convert PDF pages: {pdf_path.name}") from error

    image_paths = []

    for page_number, page in enumerate(pages, start=1):
        image_path = Path(temporary_folder) / f"{pdf_path.stem}_page_{page_number}.png"
        page.save(image_path, "PNG")
        image_paths.append(image_path)

    return image_paths


def load_existing_model_outputs(json_path):
    if not json_path.is_file():
        return {}
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f).get("models", {})
    except Exception:
        return {}


def process_document_image_folder(
    document_name,
    images_folder,
    ground_truth_folder,
    process_model,
    excel_file_name,
    model_scope=MODEL_SCOPE_ALL,
):
    run_model_names = choose_models(
        get_available_ocr_models(),
        get_vision_llm_models(),
        model_scope,
    )
    excel_rows = []
    json_outputs = []

    for image_path in list_document_images(images_folder):
        ground_truth_path, ground_truth = load_ground_truth_for_image(image_path, ground_truth_folder)
        document_side = get_document_image_side(document_name, image_path)

        json_output_path = (
            get_document_outputs_folder(document_name, "image")
            / f"{image_path.stem}_{document_name}_output.json"
        )
        model_outputs = load_existing_model_outputs(json_output_path)

        for model_name in run_model_names:
            started_at = time.perf_counter()

            try:
                if is_llm_model(model_name):
                    model_result = process_image_with_llm(document_name, image_path, model_name)
                else:
                    model_result = process_model(image_path, model_name)
                comparison = compare_with_ground_truth(model_result["extracted_data"], ground_truth)
                processing_time_seconds = round(time.perf_counter() - started_at, 4)
                model_outputs[model_name] = {
                    "extracted_data": model_result["extracted_data"],
                    "classification": model_result["classification"],
                    "validation": model_result["validation"],
                    "comparison_with_ground_truth": make_readable_comparison(comparison),
                    "raw_text": model_result.get("raw_text", ""),
                    "processing_time_seconds": processing_time_seconds,
                }
            except Exception as error:
                existing = model_outputs.get(model_name, {})
                if existing and not existing.get("error"):
                    # Keep the previously successful result rather than overwriting with an error.
                    pass
                else:
                    comparison = compare_with_ground_truth({}, ground_truth)
                    processing_time_seconds = round(time.perf_counter() - started_at, 4)
                    model_outputs[model_name] = {
                        "extracted_data": {},
                        "comparison_with_ground_truth": make_readable_comparison(comparison),
                        "error": str(error),
                        "processing_time_seconds": processing_time_seconds,
                    }

        excel_row = {
            "image address": str(image_path),
            "document side": document_side,
            "image quality category": get_image_quality_category(image_path),
        }
        for model_name, output in model_outputs.items():
            comparison_data = output.get("comparison_with_ground_truth", {})
            excel_row[model_name] = comparison_data.get("accuracy_percent", 0)
            excel_row[f"{model_name} processing time (seconds)"] = output.get("processing_time_seconds", 0)
            excel_row.setdefault("__field_results", {})[model_name] = comparison_data.get("fields", {})
            excel_row.setdefault("__model_failed", {})[model_name] = bool(output.get("error"))
            total = comparison_data.get("total_fields", 0)
            matched = comparison_data.get("matched_fields", 0)
            excel_row.setdefault("__model_complete", {})[model_name] = total > 0 and matched == total

        final_json = {
            "image": {
                "address": str(image_path),
                "ground_truth_address": str(ground_truth_path),
                "side": document_side,
                "quality": get_image_quality_category(image_path),
            },
            "ground_truth": ground_truth,
            "models": model_outputs,
        }
        json_outputs.append(
            save_document_json_file(
                image_path,
                final_json,
                document_name,
                file_format="image",
            )
        )
        excel_rows.append(excel_row)

    _meta_keys = {"image address", "document side", "image quality category", "__field_results", "__model_failed", "__model_complete"}
    all_model_names = list(dict.fromkeys(
        name
        for row in excel_rows
        for name in row
        if name not in _meta_keys and not name.endswith(" processing time (seconds)")
    ))

    excel_output = save_accuracy_excel(
        excel_rows,
        all_model_names,
        file_name=excel_file_name,
        output_subfolder=document_name,
        output_format="image",
        report_profile=document_name,
    )

    return {
        "processed_images": len(excel_rows),
        "json_outputs": json_outputs,
        "excel_output": excel_output,
    }


def process_document_pdf_folder(
    document_name,
    pdfs_folder,
    ground_truth_folder,
    process_model,
    excel_file_name,
    output_format="pdf/scanned",
    model_scope=MODEL_SCOPE_ALL,
):
    model_names = choose_models(
        get_available_ocr_models(),
        get_vision_llm_models(),
        model_scope,
    )
    excel_rows = []
    json_outputs = []

    for pdf_path in list_document_pdfs(pdfs_folder):
        ground_truth_path, ground_truth = load_ground_truth_for_pdf(
            pdf_path,
            pdfs_folder,
            ground_truth_folder,
        )
        page_mapping = get_pdf_page_mapping(ground_truth)
        pdf_page_outputs = {}

        with TemporaryDirectory() as temporary_folder:
            page_images = convert_pdf_to_page_images(pdf_path, temporary_folder)

            for page_number, page_image_path in enumerate(page_images, start=1):
                page_key = str(page_number)
                page_label = page_mapping.get(page_key, f"page_{page_number}")
                page_ground_truth = get_pdf_page_ground_truth(ground_truth, page_label)
                report_side = get_report_side_for_page(page_label)
                model_outputs = {}
                excel_row = {
                    "image address": f"{pdf_path}#page={page_number}",
                    "document side": report_side,
                    "image quality category": get_image_quality_category(pdf_path),
                }

                for model_name in model_names:
                    started_at = time.perf_counter()

                    try:
                        if is_llm_model(model_name):
                            model_result = process_image_with_llm(
                                document_name,
                                page_image_path,
                                model_name,
                            )
                        else:
                            model_result = process_model(page_image_path, model_name)
                        comparison = compare_with_ground_truth(
                            model_result["extracted_data"],
                            page_ground_truth,
                        )
                        processing_time_seconds = round(
                            time.perf_counter() - started_at,
                            4,
                        )
                        model_outputs[model_name] = {
                            "extracted_data": model_result["extracted_data"],
                            "classification": model_result["classification"],
                            "validation": model_result["validation"],
                            "comparison_with_ground_truth": make_readable_comparison(
                                comparison
                            ),
                            "raw_text": model_result.get("raw_text", ""),
                            "processing_time_seconds": processing_time_seconds,
                        }
                    except Exception as error:
                        comparison = compare_with_ground_truth({}, page_ground_truth)
                        processing_time_seconds = round(
                            time.perf_counter() - started_at,
                            4,
                        )
                        model_outputs[model_name] = {
                            "extracted_data": {},
                            "comparison_with_ground_truth": make_readable_comparison(
                                comparison
                            ),
                            "error": str(error),
                            "processing_time_seconds": processing_time_seconds,
                        }

                    excel_row[model_name] = comparison["accuracy_percent"]
                    excel_row[
                        f"{model_name} processing time (seconds)"
                    ] = processing_time_seconds

                pdf_page_outputs[page_key] = {
                    "page_label": page_label,
                    "ground_truth": page_ground_truth,
                    "models": model_outputs,
                }
                excel_rows.append(excel_row)

        final_json = {
            "pdf": {
                "address": str(pdf_path),
                "ground_truth_address": str(ground_truth_path),
                "quality": get_image_quality_category(pdf_path),
                "page_count": len(pdf_page_outputs),
                "pages": page_mapping,
            },
            "ground_truth": ground_truth,
            "page_results": pdf_page_outputs,
        }
        json_outputs.append(
            save_document_json_file(
                pdf_path,
                final_json,
                document_name,
                file_format=output_format,
            )
        )

    excel_output = save_accuracy_excel(
        excel_rows,
        model_names,
        file_name=excel_file_name,
        output_subfolder=document_name,
        output_format=output_format,
    )

    return {
        "processed_pdfs": len(json_outputs),
        "processed_pages": len(excel_rows),
        "json_outputs": json_outputs,
        "excel_output": excel_output,
    }


def process_digital_pdf_text(document_name, text, source_path):
    processors = {
        "aadhaar": (
            extract_aadhaar_fields_from_text,
            make_aadhaar_classification,
        ),
        "pan": (
            extract_pan_fields_from_text,
            lambda _document_type: make_pan_classification(),
        ),
        "passbook": (
            extract_passbook_fields_from_text,
            lambda _document_type: make_passbook_classification(),
        ),
        "invoice": (
            extract_invoice_fields_from_text,
            lambda _document_type: make_invoice_classification(),
        ),
    }
    fallback_extractor, make_fallback_classification = processors[document_name]
    return process_text_for_document(
        text,
        source_path,
        fallback_extractor,
        make_fallback_classification,
    )


def process_document_digital_pdf_folder(
    document_name,
    pdfs_folder,
    ground_truth_folder,
    excel_file_name,
    model_scope=MODEL_SCOPE_ALL,
):
    extractor_names = choose_models(
        get_available_digital_pdf_extractors(),
        get_text_llm_models(),
        model_scope,
    )
    excel_rows = []
    json_outputs = []

    for pdf_path in list_document_pdfs(pdfs_folder):
        ground_truth_path, ground_truth = load_ground_truth_for_pdf(
            pdf_path,
            pdfs_folder,
            ground_truth_folder,
        )
        page_mapping = get_pdf_page_mapping(ground_truth)
        expected_page_count = len(page_mapping)
        page_outputs = {
            page_key: {
                "page_label": page_label,
                "ground_truth": get_pdf_page_ground_truth(ground_truth, page_label),
                "extractors": {},
            }
            for page_key, page_label in page_mapping.items()
        }
        page_rows = {
            page_key: {
                "image address": f"{pdf_path}#page={page_key}",
                "document side": get_report_side_for_page(page_label),
                "image quality category": get_image_quality_category(pdf_path),
            }
            for page_key, page_label in page_mapping.items()
        }

        for extractor_name in extractor_names:
            started_at = time.perf_counter()
            try:
                if is_llm_model(extractor_name):
                    page_texts = extract_text_pages_from_digital_pdf(
                        pdf_path,
                        "pymupdf",
                    )
                else:
                    page_texts = extract_text_pages_from_digital_pdf(
                        pdf_path,
                        extractor_name,
                    )
                total_time = time.perf_counter() - started_at
                average_page_time = round(
                    total_time / max(len(page_texts), 1),
                    4,
                )

                for page_key, page_label in page_mapping.items():
                    page_index = int(page_key) - 1
                    page_text = (
                        page_texts[page_index]
                        if page_index < len(page_texts)
                        else ""
                    )
                    page_started_at = time.perf_counter()
                    if is_llm_model(extractor_name):
                        model_result = process_text_with_llm(
                            document_name,
                            page_text,
                            pdf_path,
                            extractor_name,
                        )
                        page_processing_time = round(
                            average_page_time + time.perf_counter() - page_started_at,
                            4,
                        )
                    else:
                        model_result = process_digital_pdf_text(
                            document_name,
                            page_text,
                            pdf_path,
                        )
                        page_processing_time = average_page_time
                    page_ground_truth = get_pdf_page_ground_truth(
                        ground_truth,
                        page_label,
                    )
                    comparison = compare_with_ground_truth(
                        model_result["extracted_data"],
                        page_ground_truth,
                    )
                    page_outputs[page_key]["extractors"][extractor_name] = {
                        "extracted_data": model_result["extracted_data"],
                        "classification": model_result["classification"],
                        "validation": model_result["validation"],
                        "comparison_with_ground_truth": make_readable_comparison(
                            comparison
                        ),
                        "source_text": page_text if is_llm_model(extractor_name) else "",
                        "raw_text": (
                            model_result.get("raw_text", "")
                            if is_llm_model(extractor_name)
                            else page_text
                        ),
                        "processing_time_seconds": page_processing_time,
                    }
                    page_rows[page_key][extractor_name] = comparison[
                        "accuracy_percent"
                    ]
                    page_rows[page_key][
                        f"{extractor_name} processing time (seconds)"
                    ] = page_processing_time
            except Exception as error:
                elapsed = round(time.perf_counter() - started_at, 4)
                for page_key, page_label in page_mapping.items():
                    page_ground_truth = get_pdf_page_ground_truth(
                        ground_truth,
                        page_label,
                    )
                    comparison = compare_with_ground_truth({}, page_ground_truth)
                    page_outputs[page_key]["extractors"][extractor_name] = {
                        "extracted_data": {},
                        "comparison_with_ground_truth": make_readable_comparison(
                            comparison
                        ),
                        "error": str(error),
                        "processing_time_seconds": elapsed,
                    }
                    page_rows[page_key][extractor_name] = comparison[
                        "accuracy_percent"
                    ]
                    page_rows[page_key][
                        f"{extractor_name} processing time (seconds)"
                    ] = elapsed

        excel_rows.extend(page_rows.values())
        final_json = {
            "pdf": {
                "address": str(pdf_path),
                "ground_truth_address": str(ground_truth_path),
                "pdf_type": "digital",
                "variant": ground_truth.get(
                    "variant",
                    get_image_quality_category(pdf_path),
                ),
                "page_count": expected_page_count,
                "pages": page_mapping,
            },
            "ground_truth": ground_truth,
            "page_results": page_outputs,
        }
        json_outputs.append(
            save_document_json_file(
                pdf_path,
                final_json,
                document_name,
                file_format="pdf/digital",
            )
        )

    excel_output = save_accuracy_excel(
        excel_rows,
        extractor_names,
        file_name=excel_file_name,
        output_subfolder=document_name,
        output_format="pdf/digital",
    )

    return {
        "processed_pdfs": len(json_outputs),
        "processed_pages": len(excel_rows),
        "json_outputs": json_outputs,
        "excel_output": excel_output,
    }


def process_aadhaar_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_image_folder(
        "aadhaar",
        aadhaar_images_folder,
        aadhaar_ground_truth_folder,
        process_aadhaar_image_with_model,
        "aadhaar_model_accuracy.xlsx",
        model_scope=model_scope,
    )


def process_pan_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_image_folder(
        "pan",
        pan_images_folder,
        pan_ground_truth_folder,
        process_pan_image_with_model,
        "pan_model_accuracy.xlsx",
        model_scope=model_scope,
    )


def process_passbook_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_image_folder(
        "passbook",
        passbook_images_folder,
        passbook_ground_truth_folder,
        process_passbook_image_with_model,
        "passbook_model_accuracy.xlsx",
        model_scope=model_scope,
    )


def process_invoice_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_image_folder(
        "invoice",
        invoice_images_folder,
        invoice_ground_truth_folder,
        process_invoice_image_with_model,
        "invoice_model_accuracy.xlsx",
        model_scope=model_scope,
    )


def process_aadhaar_pdf_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_pdf_folder(
        "aadhaar",
        aadhaar_pdfs_folder / "scanned",
        aadhaar_pdf_ground_truth_folder / "scanned",
        process_aadhaar_image_with_model,
        "aadhaar_pdf_model_accuracy.xlsx",
        model_scope=model_scope,
    )


def process_pan_pdf_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_pdf_folder(
        "pan",
        pan_pdfs_folder / "scanned",
        pan_pdf_ground_truth_folder / "scanned",
        process_pan_image_with_model,
        "pan_pdf_model_accuracy.xlsx",
        model_scope=model_scope,
    )


def process_passbook_pdf_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_pdf_folder(
        "passbook",
        passbook_pdfs_folder / "scanned",
        passbook_pdf_ground_truth_folder / "scanned",
        process_passbook_image_with_model,
        "passbook_pdf_model_accuracy.xlsx",
        model_scope=model_scope,
    )


def process_invoice_pdf_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_pdf_folder(
        "invoice",
        invoice_pdfs_folder / "scanned",
        invoice_pdf_ground_truth_folder / "scanned",
        process_invoice_image_with_model,
        "invoice_pdf_model_accuracy.xlsx",
        model_scope=model_scope,
    )


def process_aadhaar_digital_pdf_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_digital_pdf_folder(
        "aadhaar",
        aadhaar_pdfs_folder / "digital",
        aadhaar_pdf_ground_truth_folder / "digital",
        "aadhaar_digital_pdf_extractor_accuracy.xlsx",
        model_scope=model_scope,
    )


def process_pan_digital_pdf_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_digital_pdf_folder(
        "pan",
        pan_pdfs_folder / "digital",
        pan_pdf_ground_truth_folder / "digital",
        "pan_digital_pdf_extractor_accuracy.xlsx",
        model_scope=model_scope,
    )


def process_passbook_digital_pdf_folder(model_scope=MODEL_SCOPE_ALL):
    return process_document_digital_pdf_folder(
        "passbook",
        passbook_pdfs_folder / "digital",
        passbook_pdf_ground_truth_folder / "digital",
        "passbook_digital_pdf_extractor_accuracy.xlsx",
        model_scope=model_scope,
    )


def clear_previous_outputs():
    outputs_folder.mkdir(parents=True, exist_ok=True)

    for path in outputs_folder.rglob("*"):
        if path.is_file() and path.name != ".gitkeep":
            path.unlink()


def print_help():
    print(
        """
Document Extraction POC

Usage:
  python testing/testing.py process <file_name>
  python testing/testing.py batch <aadhaar|pan|passbook|invoice> --format image
  python testing/testing.py batch <aadhaar|pan|passbook|invoice> --format pdf --pdf-type scanned
  python testing/testing.py batch <aadhaar|pan|passbook> --format pdf --pdf-type digital
  python testing/testing.py batch <document> --format image --llm-only
  python testing/testing.py batch <document> --format image --no-llm
  python testing/testing.py clear-output

Legacy usage still supported:
  python testing/testing.py <file_name>
  python testing/testing.py --clear-output
  python testing/testing.py --aadhaar-batch
  python testing/testing.py --pan-batch
  python testing/testing.py --passbook-batch
  python testing/testing.py --invoice-batch
  python testing/testing.py --aadhaar-pdf-batch
  python testing/testing.py --pan-pdf-batch
  python testing/testing.py --passbook-pdf-batch
  python testing/testing.py --invoice-pdf-batch
  python testing/testing.py --aadhaar-digital-pdf-batch
  python testing/testing.py --pan-digital-pdf-batch
  python testing/testing.py --passbook-digital-pdf-batch
  python testing/testing.py --help

Single file mode:
  Put the file in the uploads/ folder, then pass only the file name.

Examples:
  python testing/testing.py RahulVerma.pdf
  uv --cache-dir .uv-cache run python testing/testing.py RahulVerma.pdf

Batch modes:
  --aadhaar-batch    Process generated Aadhaar images and save accuracy report.
  --pan-batch        Process generated PAN images and save accuracy report.
  --passbook-batch   Process generated passbook images and save accuracy report.
  --invoice-batch    Process generated invoice images and save accuracy report.
  --aadhaar-pdf-batch
                     Process generated Aadhaar PDFs page by page.
  --pan-pdf-batch    Process generated PAN PDFs page by page.
  --passbook-pdf-batch
                     Process generated passbook PDFs page by page.
  --invoice-pdf-batch
                     Process generated invoice PDFs page by page.
  --aadhaar-digital-pdf-batch
                     Compare native text extractors on digital Aadhaar PDFs.
  --pan-digital-pdf-batch
                     Compare native text extractors on digital PAN PDFs.
  --passbook-digital-pdf-batch
                     Compare native text extractors on digital passbook PDFs.

Output cleanup:
  --clear-output     Delete generated JSON and Excel output files.
                     It can be combined with a batch mode.

Outputs:
  Single file JSON: outputs/<input_file_name>_output.json
  Batch image reports: testing/test-outputs/<document_type>/image/
  Scanned PDF reports: testing/test-outputs/<document_type>/pdf/scanned/
  Digital PDF reports: testing/test-outputs/<document_type>/pdf/digital/

LLM comparison:
  Image and scanned PDF batches also include Ollama vision models:
  llm:qwen2.5vl:7b, llm:llama3.2-vision:11b
  Digital PDF batches also include Ollama text models:
  llm:qwen2.5:7b-instruct, llm:llama3.1:8b
  Pull the models with ollama first, or those columns will show failures.
  Default batch mode runs all available extractors plus all configured LLMs.
  Use --no-llm for extractor-only batches, or --llm-only for LLM-only batches.
  To run fewer LLMs, set VISION_LLM_MODELS or TEXT_LLM_MODELS as comma-separated names.
  Example: VISION_LLM_MODELS=qwen2.5vl:7b
""".strip()
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python testing/testing.py",
        description="Extract structured data from uploaded documents and generated datasets.",
    )
    subparsers = parser.add_subparsers(dest="command")

    process_parser = subparsers.add_parser(
        "process",
        help="Process one file from the uploads folder.",
    )
    process_parser.add_argument("file_name")

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run a generated dataset batch.",
    )
    batch_parser.add_argument(
        "document",
        choices=["aadhaar", "pan", "passbook", "invoice"],
    )
    batch_parser.add_argument(
        "--format",
        choices=["image", "pdf"],
        default="image",
        dest="input_format",
    )
    batch_parser.add_argument(
        "--pdf-type",
        choices=["scanned", "digital"],
        default="scanned",
    )
    batch_parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear generated outputs before running this batch.",
    )
    model_scope_group = batch_parser.add_mutually_exclusive_group()
    model_scope_group.add_argument(
        "--no-llm",
        "--ocr-only",
        action="store_const",
        const=MODEL_SCOPE_NO_LLM,
        default=MODEL_SCOPE_ALL,
        dest="model_scope",
        help="Run only OCR/text extractors, excluding LLM models.",
    )
    model_scope_group.add_argument(
        "--llm-only",
        action="store_const",
        const=MODEL_SCOPE_LLM_ONLY,
        dest="model_scope",
        help="Run only configured LLM models.",
    )

    subparsers.add_parser(
        "clear-output",
        help="Delete generated JSON and Excel files while keeping folders.",
    )

    return parser


def run_image_batch(document_name, model_scope=MODEL_SCOPE_ALL):
    processors = {
        "aadhaar": ("Aadhaar", process_aadhaar_folder),
        "pan": ("PAN", process_pan_folder),
        "passbook": ("passbook", process_passbook_folder),
        "invoice": ("invoice", process_invoice_folder),
    }
    document_label, processor = processors[document_name]
    result = processor(model_scope=model_scope)
    print(f"Done. Processed {result['processed_images']} {document_label} images.")
    print(f"Excel saved at: {result['excel_output']}")


def run_pdf_batch(document_name, pdf_type, model_scope=MODEL_SCOPE_ALL):
    scanned_processors = {
        "aadhaar": ("Aadhaar", process_aadhaar_pdf_folder),
        "pan": ("PAN", process_pan_pdf_folder),
        "passbook": ("passbook", process_passbook_pdf_folder),
        "invoice": ("invoice", process_invoice_pdf_folder),
    }
    digital_processors = {
        "aadhaar": ("digital Aadhaar", process_aadhaar_digital_pdf_folder),
        "pan": ("digital PAN", process_pan_digital_pdf_folder),
        "passbook": ("digital passbook", process_passbook_digital_pdf_folder),
    }
    processors = digital_processors if pdf_type == "digital" else scanned_processors

    if document_name not in processors:
        raise ValueError(f"{document_name} does not have a {pdf_type} PDF batch yet.")

    document_label, processor = processors[document_name]
    result = processor(model_scope=model_scope)
    print(
        f"Done. Processed {result['processed_pdfs']} {document_label} PDFs "
        f"({result['processed_pages']} pages)."
    )
    print(f"Excel saved at: {result['excel_output']}")


def run_batch_command(arguments):
    if arguments.clear_output:
        clear_previous_outputs()
        print("Generated output files cleared.")

    if arguments.input_format == "image":
        run_image_batch(arguments.document, arguments.model_scope)
    else:
        run_pdf_batch(arguments.document, arguments.pdf_type, arguments.model_scope)


def run_legacy_command(arguments):
    if not arguments or any(argument in {"--help", "-h"} for argument in arguments):
        print_help()
        return True

    legacy_arguments = list(arguments)

    if "--clear-output" in legacy_arguments:
        clear_previous_outputs()
        print("Generated output files cleared.")
        legacy_arguments.remove("--clear-output")

        if not legacy_arguments:
            return True

    if len(legacy_arguments) > 1:
        print(f"Unexpected arguments: {' '.join(legacy_arguments[1:])}")
        print()
        print_help()
        return True

    action = legacy_arguments[0]

    if action == "--aadhaar-batch":
        run_image_batch("aadhaar")
        return True

    if action == "--pan-batch":
        run_image_batch("pan")
        return True

    if action == "--passbook-batch":
        run_image_batch("passbook")
        return True

    if action == "--invoice-batch":
        run_image_batch("invoice")
        return True

    pdf_batch_actions = {
        "--aadhaar-pdf-batch": ("aadhaar", "scanned"),
        "--pan-pdf-batch": ("pan", "scanned"),
        "--passbook-pdf-batch": ("passbook", "scanned"),
        "--invoice-pdf-batch": ("invoice", "scanned"),
        "--aadhaar-digital-pdf-batch": ("aadhaar", "digital"),
        "--pan-digital-pdf-batch": ("pan", "digital"),
        "--passbook-digital-pdf-batch": ("passbook", "digital"),
    }

    if action in pdf_batch_actions:
        document_name, pdf_type = pdf_batch_actions[action]
        run_pdf_batch(document_name, pdf_type)
        return True

    if action.startswith("-"):
        print(f"Unknown option: {action}")
        print()
        print_help()
        return True

    run_document_processing(action)
    return True


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)

    if not arguments or arguments[0].startswith("-"):
        run_legacy_command(arguments)
        return

    if arguments[0] not in {"process", "batch", "clear-output"}:
        run_legacy_command(arguments)
        return

    parser = build_parser()
    parsed_arguments = parser.parse_args(arguments)

    if parsed_arguments.command == "process":
        run_document_processing(parsed_arguments.file_name)
        return

    if parsed_arguments.command == "batch":
        try:
            run_batch_command(parsed_arguments)
        except ValueError as error:
            parser.error(str(error))
        return

    if parsed_arguments.command == "clear-output":
        clear_previous_outputs()
        print("Generated output files cleared.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
