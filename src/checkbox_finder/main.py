import random

import cv2


HUMAN_EYE_RESULTS = {
    "checkbox_count": 42,
    "checked": 16,
    "unchecked": 26,
}
RED = (0, 0, 255)
GREEN = (0, 255, 0)


def get_random_text_position(x, y, w, h, img_width, img_height):
    # 8 possible positions around the box
    positions = [
        (x - w, y - h),  # top left
        (x + w // 2, y - h),  # top center
        (x + w, y - h),  # top right
        (x - w, y + h // 2),  # middle left
        (x + w, y + h // 2),  # middle right
        (x - w, y + h),  # bottom left
        (x + w // 2, y + h),  # bottom center
        (x + w, y + h),  # bottom right
    ]

    # Filter positions that would go outside image bounds
    valid_positions = [
        (px, py)
        for px, py in positions
        if 0 <= px <= img_width - 120 and 0 <= py <= img_height - 30
    ]

    return random.choice(valid_positions) if valid_positions else (x, y - 30)


def process_image(image_path, output_path, debug_mode: bool = False):
    # https://stackoverflow.com/a/55767996
    # 1. Obtain binary image.
    #   Load the image, grayscale, Gaussian blur, and Otsu's threshold to obtain a binary black/white image.
    print("process_image - Loading the image")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)  # TODO (test with 3,3 or 7,7)
    _ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if debug_mode:
        cv2.imwrite("1-process-image-gray.png", otsu)

    print("process_image - Preprocessing the image")
    _, binary = cv2.threshold(otsu, 128, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150)
    if debug_mode:
        cv2.imwrite("2-process-image-binary.png", binary)
        cv2.imwrite("3-process-image-edges.png", edges)

    print("process_image - Finding contours")
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if debug_mode:
        contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
        cv2.imwrite("4-process-image-contours.png", contour_image)

    # Create debug image with size annotations
    if debug_mode:
        debug_image = image.copy()

    print(f"process_image - Found {len(contours)} contours")
    algorithm_results = {
        "checkbox_count": 0,
        "checked": 0,
        "unchecked": 0,
    }
    visible_count = 0
    for idx, contour in enumerate(contours):
        contour_debug_prefix = f"{idx+1}/{len(contours)} -"
        print(f"{contour_debug_prefix} Approximating to polygon")
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        print(f"{contour_debug_prefix} Filtering for rectangular shapes")
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = w * h

            # Size-based filtering
            MIN_AREA = 100  # minimum pixel area
            MAX_AREA = 2500  # maximum pixel area

            # TODO: Fine-tune size/aspect ratio for checkboxes
            if 0.9 <= aspect_ratio <= 1.1 and MIN_AREA <= area <= MAX_AREA:
                # At this point, we consider the contour a checkbox
                visible_count += 1
                algorithm_results["checkbox_count"] += 1

                # Analyze the fill density inside the rectangle
                print(f"{contour_debug_prefix} Analyzing fill density of rectangle")
                roi = binary[y : y + h, x : x + w]
                fill_ratio = cv2.countNonZero(roi) / area

                if debug_mode:
                    height, width = debug_image.shape[:2]
                    text_x, text_y = get_random_text_position(x, y, w, h, width, height)

                    # White background for text visibility
                    cv2.rectangle(
                        debug_image,
                        (text_x, text_y),
                        (text_x + 120, text_y + 25),
                        (0, 255, 255),
                        -1,
                    )

                    # Text with box number and fill ratio
                    text = f"#{visible_count} ({fill_ratio:.2f})"
                    cv2.putText(
                        debug_image,
                        text,
                        (text_x + 5, text_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )

                print(f"{contour_debug_prefix} Classifying as filled or unfilled")
                # TODO: Optimize fill ratio threshold
                color = (
                    # Green for filled, Red for unfilled
                    GREEN
                    if fill_ratio > 0.3
                    else RED
                )
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                if color == GREEN:
                    algorithm_results["checked"] += 1
                elif color == RED:
                    algorithm_results["unchecked"] += 1
    if debug_mode:
        cv2.imwrite("5-debug_sizes.png", debug_image)

    # Save the output image
    print("process_image - Saving the output image")
    cv2.imwrite(output_path, image)
    print("process_image - Done")
    print("Human results: ", HUMAN_EYE_RESULTS)
    print("Algorithm results: ", algorithm_results)


# Instructions for running the script
if __name__ == "__main__":
    input_path = "0-input.webp"  # Replace with your input file path
    output_path = "output_highlighted.png"
    process_image(input_path, output_path, True)
    print(f"Processed image saved to {output_path}")
