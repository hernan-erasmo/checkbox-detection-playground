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
    """
    Convert to grayscale, Gaussian blur, Otsu's threshold
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if debug_mode:
        cv2.imwrite("1-converted-to-grayscale.png", thresh)
    return thresh


def find_contours(processed_image: MatLike, debug_mode: bool = False) -> list:
    """
    Find contours and filter using contour area filtering to remove noise
    """
    cnts, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[
        -2:
    ]
    AREA_THRESHOLD = 10
    contours_under_threshold = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < AREA_THRESHOLD:
            contours_under_threshold += 1
            cv2.drawContours(processed_image, [c], -1, 0, -1)
    if debug_mode:
        print(
            f"step 2) contours found / under_threshold: {len(cnts)}/{contours_under_threshold}"
        )
        cv2.imwrite("2-found-contours.png", processed_image)
    return cnts


def repair_image(processed_image: MatLike, debug_mode: bool = False) -> MatLike:
    """
    Repair checkbox horizontal and vertical walls
    """
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    repaired_horizontal = cv2.morphologyEx(
        processed_image, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1
    )
    repaired_vertical = cv2.morphologyEx(
        repaired_horizontal, cv2.MORPH_CLOSE, vertical_kernel, iterations=1
    )
    if debug_mode:
        cv2.imwrite("3-repaired-image.png", repaired_vertical)
    return repaired_vertical


def find_checkboxes_from_image(
    repaired_image: MatLike, debug_mode: bool = False
) -> list:
    """
    Detect checkboxes using shape approximation and aspect ratio filtering
    """
    checkbox_contours = []
    cnts, _ = cv2.findContours(
        repaired_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2:]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.035 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if len(approx) == 4 and (aspect_ratio >= 0.8 and aspect_ratio <= 1.2):
            checkbox_contours.append(c)
    if debug_mode:
        print(
            f"step 4) iterated over {len(cnts)} contours, found {len(checkbox_contours)} checkboxes"
        )
    return checkbox_contours


def process_image(image_path: str, output_path: str, debug_mode: bool = False):
    """
    This is the main function, and it follows the steps of an algorithm described
    at https://stackoverflow.com/a/55767996
    """

    # Load image
    image = cv2.imread(image_path)
    results_image_copy = image.copy()

    # Step 1
    thresh = convert_to_grayscale(image, debug_mode)

    # Step 2
    # cnts = find_contours(thresh, debug_mode)

    # Step 3
    repaired_image = repair_image(thresh, debug_mode)

    # Step 4
    checkboxes = find_checkboxes_from_image(repaired_image, debug_mode)

    # Step 5 (print results)
    for check in checkboxes:
        x, y, w, h = cv2.boundingRect(check)
        cv2.rectangle(results_image_copy, (x, y), (x + w, y + h), (36, 255, 12), 3)
    cv2.imwrite(output_path, results_image_copy)


if __name__ == "__main__":
    input_path = "0-input.webp"  # Replace with your input file path
    output_path = "9-output.png"
    debugging_enabled = True
    process_image(input_path, output_path, debugging_enabled)
