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

using namespace godot;

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
    for (auto i = 0; i < contours.size(); i++)
    {
        // Approximate found cotours more efficently using polygonal curves
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true) * 0.02, true);

        // Filter for contours with 4 Points (we are only interested in boxes)
        if (approx.size() != 4)
        {
            continue;
        }

        // Create bounding boxes 
        cv::Rect bounding_box = cv::boundingRect(approx);
        
        // Filter bounding boxes that are to big or small
        int max_width = frame_raw.cols - bounding_box_max_width;
        int max_height = frame_raw.rows - bounding_box_max_height;
        if (!(bounding_box.width >= bounding_box_min_width && bounding_box.height >= bounding_box_min_height && bounding_box.width <= max_width && bounding_box.height <= max_height))
        {
            continue;
        }

        // Remove non convex contours
        if (!cv::isContourConvex(approx))
        {
            continue;
        }

        bool isClosed = true;
        auto color = cv::Scalar(255, 0, 0);
        cv::polylines(frame_overlay, approx, isClosed, color);

        // Subdivide edge
        auto edge_a = cv::Point2f(approx[1]) - cv::Point2f(approx[0]);
        auto edge_b = cv::Point2f(approx[3]) - cv::Point2f(approx[0]);
        auto edge_c = cv::Point2f(approx[1]) - cv::Point2f(approx[2]);
        auto edge_d = cv::Point2f(approx[3]) - cv::Point2f(approx[2]);
        auto circle_color = cv::Scalar(0, 255, 0);
        std::vector<cv::Point> subdivision_vector;
        for (auto j = 1; j < 7; j++) {
            std::vector<cv::Point2f> buf;
            buf.insert(buf.begin(), cv::Point2f(approx[0]) + edge_a * (j / 7.0));
            buf.insert(buf.begin(), cv::Point2f(approx[0]) + edge_b * (j / 7.0));
            buf.insert(buf.begin(), cv::Point2f(approx[2]) + edge_c * (j / 7.0));
            buf.insert(buf.begin(), cv::Point2f(approx[2]) + edge_d * (j / 7.0));
            for (auto k = 0; k < buf.size(); k++) {
                cv::circle(frame_overlay, buf[k], 3, circle_color);
            }
        }
        //cv::Sobel()
    }
    return;
}

void CVCamera::flip(bool flip_lr, bool flip_ud)
{
    this->flip_lr = flip_lr;
    this->flip_ud = flip_ud;
}

void calculateStripDimensions(double dx, double dy, StripDimensions &st)
{

}

int subpixSampleSafe(const cv::Mat &pSrc, const cv::Point2f &p)
{
    int x = int( floorf ( p.x ) );
    int y = int( floorf ( p.y ) );
    if ( x < 0 || x >= pSrc.cols - 1 || y < 0 || y >= pSrc.rows - 1 )
        return 127; // TODO Why is 127 returned?
    // ( p.x - floorf ( p.x ) ) <=> Calculate the ratio of normal pixel area covered by the subpixel
    int dx = int ( 256 * ( p.x - floorf ( p.x ) ) );    // TODO Why scale with 256?
    int dy = int ( 256 * ( p.y - floorf ( p.y ) ) );
    auto i = ( unsigned char* ) ( ( pSrc.data + y * pSrc.step ) + x );
    int a = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) ) >> 8 );
    i += pSrc.step;
    int b = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) ) >> 8 );
    return a + ( ( dy * ( b - a) ) >> 8 );
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