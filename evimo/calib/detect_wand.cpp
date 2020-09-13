#include "detect_wand.h"

using namespace wand;

int main (int argc, char** argv) {
    std::cout << argv[1] << "\n";

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    auto keypoints = get_blobs(image);

    std::cout << "Keypoints:\n";
    for (auto &p : keypoints) {
        std::cout << "\t" << p.pt.x << " " << p.pt.y << "\n";
    }

    // Coordinates of red leds:
    std::valarray<std::valarray<float>> wand_3d_mapping = {{30.576542, -150.730270, -45.588951},
                                                           {-45.614960, -18.425253, 2.359259},
                                                           {-83.571022, 47.522499, 26.421692},
                                                           {64.297485, 39.157578, 6.395612},
                                                           {169.457520, 97.256927, 11.720362}};

    auto idx = find_all_3lines(keypoints, 0.2);
    std::cout << "Lines:\n";
    for (auto &line : idx) {
        std::cout << "\t" << line[0] << " " << line[1] << " " << line[2] << "\n";
    }


    auto wand_points = detect_wand_internal(keypoints, idx, wand_3d_mapping, 0.5, 0.5, 0.5);
    if (wand_points.size() == 0)
        std::cout << "Failed detection!\n";

    std::cout << "Points:\n";
    for (auto &point : wand_points) {
        std::cout << "\t" << point.x << " " << point.y << "\n";
    }


    wand_points = detect_wand(image, 0.5, 0.5, 0.5);

    auto im_with_wand = draw_wand(image, wand_points);
    cv::imshow("wand", im_with_wand);
    cv::waitKey(0);

    return 0;
};
