/*
Contribution credits:
Leonard Keil <leonard.keil@tum.de>
Jannik Waldraff <jannik.waldraff@tum.de>
Sandro Weber <sandro.weber@tum.de>
*/

#include "CVCamera.h"

#include <opencv2/imgproc.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <bitset>
#include <opencv2/imgproc.hpp>

using namespace godot;
using namespace std;

typedef cv::Vec<uint8_t, 4> Pixel;

void CVCamera::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("open"), &CVCamera::open);
    ClassDB::bind_method(D_METHOD("open_file"), &CVCamera::open_file);
    ClassDB::bind_method(D_METHOD("close"), &CVCamera::close);
    ClassDB::bind_method(D_METHOD("get_image"), &CVCamera::get_image);
    ClassDB::bind_method(D_METHOD("flip"), &CVCamera::flip);
    ClassDB::bind_method(D_METHOD("get_framerate"), &CVCamera::get_framerate);
    ClassDB::bind_method(D_METHOD("get_threshold_image"), &CVCamera::get_threshold_image);
    ClassDB::bind_method(D_METHOD("get_greyscale_image"), &CVCamera::get_greyscale_image);
    ClassDB::bind_method(D_METHOD("update_threshold_value"), &CVCamera::update_threshold_value);
    ClassDB::bind_method(D_METHOD("update_thres_type"), &CVCamera::update_thres_type);
    ClassDB::bind_method(D_METHOD("update_threshold_max_value"), &CVCamera::update_threshold_max_value);
    ClassDB::bind_method(D_METHOD("update_threshold_adaptive_type"), &CVCamera::update_threshold_adaptive_type);  
    ClassDB::bind_method(D_METHOD("update_threshold_blocksize"), &CVCamera::update_threshold_blocksize);  
    ClassDB::bind_method(D_METHOD("update_threshold_c"), &CVCamera::update_threshold_c);
    ClassDB::bind_method(D_METHOD("find_rectangles"), &CVCamera::find_rectangles);
    ClassDB::bind_method(D_METHOD("get_overlay_image"), &CVCamera::get_overlay_image);
    ClassDB::bind_method(D_METHOD("set_bounding_box_min_width"), &CVCamera::set_bounding_box_min_width);
    ClassDB::bind_method(D_METHOD("get_bounding_box_min_width"), &CVCamera::get_bounding_box_min_width);
    ClassDB::bind_method(D_METHOD("frame_forward"), &CVCamera::frame_forward);
    ClassDB::bind_method(D_METHOD("frame_backward"), &CVCamera::frame_backward);
    ClassDB::bind_method(D_METHOD("get_current_frame"), &CVCamera::get_current_frame);
    ClassDB::bind_method(D_METHOD("get_frame_count"), &CVCamera::get_frame_count);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "bounding_box_min_width"), "set_bounding_box_min_width", "get_bounding_box_min_width");
}

CVCamera::CVCamera()
{
    last_update_frame = -1;
    video_file_playback = false;
}

CVCamera::~CVCamera()
{
    close();
}

void CVCamera::open(int device, int width = 1920, int height = 1080)
{
    capture.open(device);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    if (!capture.isOpened())
    {
        capture.release();
        UtilityFunctions::push_error("Couldn't open camera.");
    }
    framerate = capture.get(cv::CAP_PROP_FPS);
}

void CVCamera::open_file(String path)
{
    const cv::String pathStr(path.utf8());
    capture.open(pathStr);
    if (!capture.isOpened())
    {
        capture.release();
        UtilityFunctions::push_error("Couldn't open file.");
    }
    framerate = capture.get(cv::CAP_PROP_FPS);
    video_file_playback = true;
}

void CVCamera::close()
{
    capture.release();
}

void CVCamera::update_frame()
{
    // Only update the frame once per godot process frame
    uint64_t current_frame = Engine::get_singleton()->get_process_frames();
    if (current_frame == last_update_frame)
    {
        return;
    }
    last_update_frame = current_frame;

    // Read the frame from the camera
    bool success = capture.read(frame_raw);

    if (!success && video_file_playback)
    {
        printf("looping video\n");

        capture.set(cv::CAP_PROP_POS_FRAMES, 0);
        success = capture.read(frame_raw);
    }

    if (!success)
    {
        printf("Error: Could not read frame\n");
        return;
    }

    if (flip_lr || flip_ud)
    {
        int code = flip_lr ? (flip_ud ? -1 : 1) : 0;
        cv::flip(frame_raw, frame_raw, code);
    }

    cv::cvtColor(frame_raw, frame_rgb, cv::COLOR_BGR2RGB);
    cv::cvtColor(frame_rgb, frame_gray, cv::COLOR_RGB2GRAY);
    // TODO Wtf are we doing here?
    frame_overlay = cv::Mat::zeros(frame_raw.size(), CV_8UC4);
    cv::cvtColor(frame_overlay, frame_overlay, cv::COLOR_BGRA2RGBA);
}

void CVCamera::frame_forward()
{
    auto current_frame = capture.get(cv::CAP_PROP_POS_FRAMES);
    capture.set(cv::CAP_PROP_POS_FRAMES, current_frame);
}

void CVCamera::frame_backward()
{
    auto current_frame = capture.get(cv::CAP_PROP_POS_FRAMES);
    capture.set(cv::CAP_PROP_POS_FRAMES, current_frame - 1);
}

double CVCamera::get_current_frame()
{
    return capture.get(cv::CAP_PROP_POS_FRAMES);
}

double CVCamera::get_frame_count()
{
    return capture.get(cv::CAP_PROP_FRAME_COUNT);
}

Ref<Image> CVCamera::mat_to_image(cv::Mat mat)
{
    cv::Mat image_mat;
    if (mat.channels() == 1)
    {
        cv::cvtColor(mat, image_mat, cv::COLOR_GRAY2RGB);
    }
    else if (mat.channels() == 4)
    {
        // Turn Pixels alpha value opaque, where there is anything but black
        image_mat = mat;
        image_mat.forEach<Pixel>([](Pixel &p, const int *position) -> void
                                 {
            if (p[0] > 0 || p[1] > 0 || p[2] > 0)
            {
                p[3] = 255;
            } });
    }
    else
    {
        image_mat = mat;
    }

    int sizear = image_mat.cols * image_mat.rows * image_mat.channels();

    PackedByteArray bytes;
    bytes.resize(sizear);
    memcpy(bytes.ptrw(), image_mat.data, sizear);

    Ref<Image> image;
    if (image_mat.channels() == 4)
    {
        image = Image::create_from_data(image_mat.cols, image_mat.rows, false, Image::Format::FORMAT_RGBA8, bytes);
    }
    else
    {
        image = Image::create_from_data(image_mat.cols, image_mat.rows, false, Image::Format::FORMAT_RGB8, bytes);
    }
    return image;
}

Ref<Image> CVCamera::get_image()
{
    update_frame();

    return mat_to_image(frame_rgb);
}

Ref<Image> CVCamera::get_greyscale_image()
{
    update_frame();
    return mat_to_image(frame_gray);
}

Ref<Image> CVCamera::get_threshold_image()
{
    update_frame();
    if (threshold_value <= 0.0) {
        // Adaptive threshold
        // TODO Fix threshold type to limit values to applicable range
        cv::adaptiveThreshold(frame_gray, frame_threshold, threshold_max_value, threshold_adaptive_type, threshold_simple_type, threshold_blocksize, threshold_c);
    } else {
        // Use simple threshold for thresh values: 0 < thresh < 255
        cv::threshold(frame_gray, frame_threshold, threshold_value, threshold_max_value, threshold_simple_type);
    }
    return mat_to_image(frame_threshold);
}

Ref<Image> CVCamera::get_overlay_image()
{
    update_frame();
    return mat_to_image(frame_overlay);
}

void CVCamera::find_rectangles()
{
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(frame_threshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> rectangles;

    for (auto i = 0; i < contours.size(); i++)
    {
        // Approximate found cotours more efficently using polygonal curves
        std::vector<cv::Point> approximation;
        cv::approxPolyDP(contours[i], approximation, cv::arcLength(contours[i], true) * 0.02, true);

        // Filter for contours with 4 Points (we are only interested in boxes)
        if (approximation.size() != 4)
        {
            continue;
        }

        // Create bounding boxes 
        cv::Rect bounding_box = cv::boundingRect(approximation);
        
        // Filter bounding boxes that are to big or small
        int max_width = frame_raw.cols - bounding_box_max_width;
        int max_height = frame_raw.rows - bounding_box_max_height;
        if (!(bounding_box.width >= bounding_box_min_width && bounding_box.height >= bounding_box_min_height && bounding_box.width <= max_width && bounding_box.height <= max_height))
        {
            continue;
        }

        // Remove non convex contours
        if (!cv::isContourConvex(approximation))
        {
            continue;
        }

        rectangles.insert(rectangles.end(), approximation);
        bool isClosed = true;
        auto color = cv::Scalar(255, 0, 0);
        cv::polylines(frame_overlay, approximation, isClosed, color);

        // This array saves the edge lines create from subpixel accurate edge points
        float subpix_line_params[16];   // Stor parameters for lines fitted trough subpixel accurate edge points
        cv::Mat lineParamsMat(cv::Size(4, 4), CV_32F, subpix_line_params);

        // Subdivide edge
        auto circle_color = cv::Scalar(0, 255, 0);

        // Iterate over Corners/Edges of the boxes
        for (int edge_index = 0; edge_index < approximation.size(); edge_index++)
        {
            cv::Point line = approximation[(edge_index + 1) % 4] - approximation[edge_index];
            cv::Point2f subpix_edge_points[6];
            // Iterate 7 times to subdivide the edge
            for (int subdiv_index = 0; subdiv_index < 7; subdiv_index++) {
                std::vector<cv::Point2f> buf;
                cv::Point subdivision_edge_point = approximation[edge_index] + (line * (subdiv_index / 7.0));

                // Draw markers
                cv::circle(frame_overlay, subdivision_edge_point, 3, circle_color);

                // Skip the corner
                if (subdiv_index == 0)
                {
                    continue;
                }

                calculateSubpixEdgePoint(subdivision_edge_point, line, subpix_edge_points[subdiv_index-1], true);
            }
            auto point_mat = cv::Mat( cv::Size(1, 6), CV_32FC2, subpix_edge_points);    // Contains 6 Subdivision edge points
            cv::fitLine ( point_mat, lineParamsMat.col(edge_index), cv::DIST_L2, 0, 0.01, 0.01);
            std::array<cv::Point2f, 4> subpixCorners = calculateSubpixCorners(subpix_line_params, true);
        }
    }
}

void CVCamera::flip(bool flip_lr, bool flip_ud)
{
    this->flip_lr = flip_lr;
    this->flip_ud = flip_ud;
}

std::array<cv::Point2f, 4> CVCamera::calculateSubpixCorners(float subpix_line_params[16], bool draw_on_overlay = true)
{
    std::array<cv::Point2f, 4> subpix_corners;
    for (auto i = 0; i < 16; i++)
    {
        //UtilityFunctions::push_warning(subpix_line_params[i]);
    }

    // Calculate the intersection points of both lines
    for (int i = 0; i < 4; ++i)
    {
        // Go through the corners of the rectangle, 3 -> 0
        int j = (i + 1) % 4;

        double x0, x1, y0, y1, u0, u1, v0, v1;

        // We have to jump through the 4x4 matrix, meaning the next value for the wanted line is in the next row -> +4
        // g: Point + d*Vector
        // g1 = (x0,y0) + scalar0*(u0,v0) == g2 = (x1,y1) + scalar1*(u1,v1)
        x0 = subpix_line_params[i + 8];
        y0 = subpix_line_params[i + 12];
        x1 = subpix_line_params[j + 8];
        y1 = subpix_line_params[j + 12];

        // Direction vector
        u0 = subpix_line_params[i];
        v0 = subpix_line_params[i + 4];
        u1 = subpix_line_params[j];
        v1 = subpix_line_params[j + 4];

        // (x|y) = p + s * vec --> Vector Equation

        // (x|y) = p + (Ds / D) * vec

        // p0.x = x0; p0.y = y0; vec0.x= u0; vec0.y=v0;
        // p0 + s0 * vec0 = p1 + s1 * vec1
        // p0-p1 = vec(-vec0 vec1) * vec(s0 s1)

        // s0 = Ds0 / D (see cramer's rule)
        // s1 = Ds1 / D (see cramer's rule)
        // Ds0 = -(x0-x1)v1 + (y0-y1)u1 --> You need to just calculate one, here Ds0

        // (x|y) = (p * D / D) + (Ds * vec / D)
        // (x|y) = (p * D + Ds * vec) / D

        // x0 * D + Ds0 * u0 / D    or   x1 * D + Ds1 * u1 / D     --> a / D
        // y0 * D + Ds0 * v0 / D    or   y1 * D + Ds1 * v1 / D     --> b / D

        // (x|y) = a / c;

        // Cramer's rule
        // 2 unknown a,b -> Equation system
        double a = x1 * u0 * v1 - y1 * u0 * u1 - x0 * u1 * v0 + y0 * u0 * u1;
        double b = -x0 * v0 * v1 + y0 * u0 * v1 + x1 * v0 * v1 - y1 * v0 * u1;

        // Calculate the cross product to check if both direction vectors are parallel -> = 0
        // c -> Determinant = 0 -> linear dependent -> the direction vectors are parallel -> No division with 0
        double c = v1 * u0 - v0 * u1;
        if (fabs(c) < 0.001)
        {
            std::cout << "lines parallel" << std::endl;
            continue;
        }

        // We have checked for parallelism of the direction vectors
        // -> Cramer's rule, now divide through the main determinant
        a /= c;
        b /= c;

        // Exact corner
        subpix_corners[i].x = a;
        subpix_corners[i].y = b;

        if (draw_on_overlay)
        {
            cv::Point point_draw;
            point_draw.x = (int)subpix_corners[i].x;
            point_draw.y = (int)subpix_corners[i].y;

            circle(frame_overlay, point_draw, 5, CV_RGB(128, 0, 128), -1);
        }
    } // End of the loop to extract the exact corners

    return subpix_corners;
}

cv::Mat CVCamera::calculateStripDimensions(double dx, double dy, StripDimensions &st)
{
    // Norm (euclidean distance) from the direction vector is the length (derived from the Pythagoras Theorem)
    double diffLength = sqrt(dx * dx + dy * dy);

    // Length proportional to the marker size
    st.stripLength = (int)(0.8 * diffLength);

    if (st.stripLength < 5)
        st.stripLength = 5;

    // Make stripeLength odd (because of the shift in nStop), Example 6: both sides of the strip must have the same length XXXOXXX
    // st.stripeLength |= 1;
    if (st.stripLength % 2 == 0)
        st.stripLength++;

    // E.g. stripeLength = 5 --> from -2 to 2: Shift -> half top, the other half bottom
    // st.nStop = st.stripeLength >> 1;
    st.nStop = st.stripLength / 2;
    st.nStart = -st.nStop;

    cv::Size stripeSize;

    // Sample a strip of width 3 pixels
    stripeSize.width = 3;
    stripeSize.height = st.stripLength;

    // Normalized direction vector
    st.stripeVecX.x = dx / diffLength;
    st.stripeVecX.y = dy / diffLength;

    // Normalized perpendicular direction vector (rotated 90  clockwise, rotation matrix)
    st.stripeVecY.x = st.stripeVecX.y;
    st.stripeVecY.y = -st.stripeVecX.x;

    // 8 bit unsigned char with 1 channel, gray
    return cv::Mat(stripeSize, CV_8UC1);
}

void CVCamera::computeStrip(cv::Point *centerPoint, StripDimensions *strip, cv::Mat *outImagePixelStrip, bool drawOnOverlay = false)
{
    // Iterate over width (3 pixels)
    for (int m = -1; m <= 1; m++)
    {
        for (int n = strip->nStart; n <= strip->nStop; n++)
        {
            cv::Point2f subPixel;

            // m -> going over the 3 pixel thickness of the stripe, n -> over the length of the stripe, direction comes from the orthogonal vector in st
            // Going from bottom to top and defining the pixel coordinate for each pixel belonging to the stripe
            subPixel.x = (double)centerPoint->x + ((double)m * strip->stripeVecX.x) + ((double)n * strip->stripeVecY.x);
            subPixel.y = (double)centerPoint->y + ((double)m * strip->stripeVecX.y) + ((double)n * strip->stripeVecY.y);

            if (drawOnOverlay)
            {
                // Just for markings in the image!
                cv::Point p2;
                p2.x = (int)subPixel.x;
                p2.y = (int)subPixel.y;

                cv::circle(frame_overlay, p2, 1, CV_RGB(255, 0, 255), -1);
            }

            // Combined Intensity of the subpixel
            int pixelIntensity = subpixSampleSafe(frame_gray, subPixel);
            // int pixelIntensity = (((m+1)+n) % 2) * 255; // TEST

            // Converte from index to pixel coordinate
            // m (Column, real) -> -1,0,1 but we need to map to 0,1,2 -> add 1 to 0..2
            int w = m + 1;

            // n (Row, real) -> add stripelenght >> 1 to shift to 0..stripeLength
            // n=0 -> -length/2, n=length/2 -> 0 ........ + length/2
            int h = n + (strip->stripLength >> 1);

            // Set pointer to correct position and safe subpixel intensity
            outImagePixelStrip->at<uchar>(h, w) = (uchar)pixelIntensity;
        }
    }
}

void CVCamera::calculateSubpixEdgePoint(cv::Point subdivision_edge_point, cv::Point line, cv::Point2f &out_subpix_edge_point, bool draw_on_overlay = false)
{
    // Compute Strip
    cv::Point point_approx_edge;
    point_approx_edge.x = (int)subdivision_edge_point.x;
    point_approx_edge.y = (int)subdivision_edge_point.y;

    StripDimensions strip_dimensions;
    cv::Mat image_pixel_strip = calculateStripDimensions(line.x / 7, line.y / 7, strip_dimensions);
    computeStrip(&point_approx_edge, &strip_dimensions, &image_pixel_strip, draw_on_overlay);

    // Sobel over the y direction
    cv::Mat sobel_gradient_y;
    cv::Sobel(image_pixel_strip, sobel_gradient_y, CV_8UC1, 0, 1);

    // Finding the max value
    double max_intensity = -1;
    int max_intensity_index = 0;
    for (int n = 0; n < strip_dimensions.stripLength - 2; ++n)
    {
        if (sobel_gradient_y.at<uchar>(n, 1) > max_intensity)
        {
            max_intensity = sobel_gradient_y.at<uchar>(n, 1);
            max_intensity_index = n;
        }
    }

    // f(x) slide 7 -> y0 .. y1 .. y2
    double y0, y1, y2;

    // Point before and after
    unsigned int max1 = max_intensity_index - 1, max2 = max_intensity_index + 1;

    // If the index is at the border we are out of the stripe, then we will take 0
    y0 = (max_intensity_index <= 0) ? 0 : sobel_gradient_y.at<uchar>(max1, 1);
    y1 = sobel_gradient_y.at<uchar>(max_intensity_index, 1);
    // If we are going out of the array of the sobel values
    y2 = (max_intensity_index >= strip_dimensions.stripLength - 3) ? 0 : sobel_gradient_y.at<uchar>(max2, 1);

    // Formula for calculating the x-coordinate of the vertex of a parabola, given 3 points with equal distances
    // (xv means the x value of the vertex, d the distance between the points):
    // xv = x1 + (d / 2) * (y2 - y0)/(2*y1 - y0 - y2)

    // d = 1 because of the normalization and x1 will be added later
    double pos = (y2 - y0) / (4 * y1 - 2 * y0 - 2 * y2);

    // What happens when there is no solution -> /0 or Number == other Number
    // If the found pos is not a number -> there is no solution
    if (isnan(pos))
    {
        return;
    }

    // Exact point with subpixel accuracy
    cv::Point2d edge_center_subpix;

    // Where is the edge (max gradient) in the picture?
    int max_index_shift = max_intensity_index - (strip_dimensions.stripLength >> 1);

    // Find the original edgepoint -> Is the pixel point at the top or bottom?
    edge_center_subpix.x = (double)point_approx_edge.x + (((double)max_index_shift + pos) * strip_dimensions.stripeVecY.x);
    edge_center_subpix.y = (double)point_approx_edge.y + (((double)max_index_shift + pos) * strip_dimensions.stripeVecY.y);

    // Highlight the subpixel with blue color
    if (draw_on_overlay)
    {
        cv::circle(frame_overlay, edge_center_subpix, 2, CV_RGB(255, 0, 0), -1);
    }

    // Save point (has to be k-1 as we only have an array of size 6 but loop through 7 points)
    out_subpix_edge_point = edge_center_subpix;
}

int CVCamera::subpixSampleSafe(const cv::Mat &pSrc, const cv::Point2f &p)
{
    int x = int(floorf(p.x));
    int y = int(floorf(p.y));
    if (x < 0 || x >= pSrc.cols - 1 || y < 0 || y >= pSrc.rows - 1)
        return 127;
    int dx = int(256 * (p.x - floorf(p.x)));
    int dy = int(256 * (p.y - floorf(p.y)));
    unsigned char *i = (unsigned char *)((pSrc.data + y * pSrc.step) + x);
    int a = i[0] + ((dx * (i[1] - i[0])) >> 8);
    i += pSrc.step;
    int b = i[0] + ((dx * (i[1] - i[0])) >> 8);
    return a + ((dy * (b - a)) >> 8);
}

void CVCamera::update_threshold_value(double thres_val)
{
    this->threshold_value = thres_val;
}

void CVCamera::update_threshold_max_value(double max_val)
{
    this->threshold_max_value = max_val;
}

void CVCamera::update_thres_type(int type)
{
    this->threshold_simple_type = type;
}

void CVCamera::update_threshold_adaptive_type(int type)
{
    this->threshold_adaptive_type = type;
}

void CVCamera::update_threshold_blocksize(float blocksize)
{
    this->threshold_blocksize = (int) blocksize;
}

void CVCamera::update_threshold_c(double c)
{
    this->threshold_c = c;
}
 
double CVCamera::get_framerate() const
{
    return framerate;
}