import cv2
from cv2.typing import MatLike


HUMAN_EYE_RESULTS = {
    "checkbox_count": 42,
    "checked": 16,
    "unchecked": 26,
}
RED = (0, 0, 255)
GREEN = (0, 255, 0)


def convert_to_grayscale(image: MatLike, debug_mode: bool = False) -> MatLike:
    """Convert to grayscale with enhanced line detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply unsharp masking for sharpening
    gaussian = cv2.GaussianBlur(gray, (5, 5), 1.5)
    sharpened = cv2.addWeighted(gray, 2.2, gaussian, -1.2, 0)

    # Optimized adaptive threshold parameters
    thresh = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        13,  # Optimal block size
        3,  # Optimal C value
    )

    if debug_mode:
        cv2.imwrite("1a-sharpened.png", sharpened)
        cv2.imwrite("1b-adaptive-thresh.png", thresh)

    return thresh


def calculate_overlap(box1, box2):
    """Calculate IoU (Intersection over Union) of two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2

    return intersection / float(box1_area + box2_area - intersection)


def detect_checkbox_contours(thresh: MatLike, debug_mode: bool = False) -> list:
    """Detect potential checkbox contours from thresholded image"""
    if debug_mode:
        cv2.imwrite("2a-thresh-input.png", thresh)

    # Find contours using TREE mode
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Store candidates with their properties
    candidates = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)

            if 0.8 <= aspect_ratio <= 1.2 and 100 <= area <= 2500:
                candidates.append(
                    {
                        "contour": contour,
                        "box": (x, y, w, h),
                        "aspect_ratio": abs(
                            1 - aspect_ratio
                        ),  # Distance from perfect square
                        "area": area,
                    }
                )

    # Remove overlapping boxes
    final_candidates = []
    while candidates:
        best = candidates.pop(0)

        # Remove any remaining candidates that overlap significantly with best
        candidates = [
            c
            for c in candidates
            if calculate_overlap(best["box"], c["box"])
            < 0.3  # Adjust threshold as needed
        ]

        final_candidates.append(best)

    # Debug visualization
    if debug_mode:
        debug_filtered = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
        for idx, candidate in enumerate(final_candidates):
            contour = candidate["contour"]
            x, y, w, h = candidate["box"]
            cv2.drawContours(debug_filtered, [contour], -1, (0, 255, 0), 2)
            cv2.putText(
                debug_filtered,
                f"#{idx}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        cv2.imwrite("2b-contours-filtered.png", debug_filtered)
        print(f"Found {len(final_candidates)} unique checkboxes")

    return [c["contour"] for c in final_candidates]


def process_image(image_path: str, output_path: str, debug_mode: bool = False):
    """
    This is the main function, and it follows the steps of an algorithm described
    at https://stackoverflow.com/a/55767996
    """

    # Load image
    image = cv2.imread(image_path)
    _results = image.copy()
    # results_from_repaired = image.copy()
    # results_from_contours = image.copy()

    # Step 1
    thresh = convert_to_grayscale(image, debug_mode)

    # Step 2
    _cnts = detect_checkbox_contours(thresh, debug_mode)
    # cnts = find_contours(thresh, debug_mode)

    # Step 3
    # repaired_image = repair_image(thresh, debug_mode)

    # Step 4
    # checkboxes_from_image = find_checkboxes_from_image(repaired_image, debug_mode)
    # checkboxes_from_contours = find_checkboxes_from_contours(cnts, debug_mode)

    # Step 5 (print results from repaired)
    # for check in checkboxes_from_image:
    #    x, y, w, h = cv2.boundingRect(check)
    #    cv2.rectangle(results_from_repaired, (x, y), (x + w, y + h), (36, 255, 12), 3)
    # cv2.imwrite("9-output-with-repaired.png", results_from_repaired)

    # Step 5 (print results from contours)
    # for check in checkboxes_from_contours:
    #    x, y, w, h = cv2.boundingRect(check)
    #    cv2.rectangle(results_from_contours, (x, y), (x + w, y + h), (36, 255, 12), 3)
    # cv2.imwrite("9-output-with-contours.png", results_from_contours)


if __name__ == "__main__":
    input_path = "0-input.webp"  # Replace with your input file path
    output_path = "9-output.png"
    debugging_enabled = True
    process_image(input_path, output_path, debugging_enabled)
