import cv2


HUMAN_EYE_RESULTS = {
    "checkbox_count": 42,
    "checked": 16,
    "unchecked": 26,
}
RED = (0, 0, 255)
GREEN = (0, 255, 0)


def process_image(image_path, output_path, debug_mode: bool = False):
    print("process_image - Loading the image")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug_mode:
        cv2.imwrite("1-process-image-gray.png", gray)

    print("process_image - Preprocessing the image")
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150)
    if debug_mode:
        cv2.imwrite("2-process-image-binary.png", binary)
        cv2.imwrite("3-process-image-edges.png", edges)

    print("process_image - Finding contours")
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if debug_mode:
        contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
        cv2.imwrite("4-process-image-contours.png", contour_image)

    print(f"process_image - Found {len(contours)} contours")
    algorithm_results = {
        "checkbox_count": 0,
        "checked": 0,
        "unchecked": 0,
    }
    for idx, contour in enumerate(contours):
        contour_debug_prefix = f"{idx+1}/{len(contours)} -"
        print(f"{contour_debug_prefix} Approximating to polygon")
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        print(f"{contour_debug_prefix} Filtering for rectangular shapes")
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # TODO: Fine-tune size/aspect ratio for checkboxes
            if 0.9 <= aspect_ratio <= 1.1 and 10 <= w <= 50:
                # At this point, we consider the contour a checkbox
                algorithm_results["checkbox_count"] += 1

                # Analyze the fill density inside the rectangle
                print(f"{contour_debug_prefix} Analyzing fill density of rectangle")
                roi = binary[y : y + h, x : x + w]
                fill_ratio = cv2.countNonZero(roi) / (w * h)

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
