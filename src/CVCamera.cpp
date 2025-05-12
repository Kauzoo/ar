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
    ClassDB::bind_method(D_METHOD("update_threshold_values"), &CVCamera::update_threshold_values);
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

Ref<Image> CVCamera::get_threshold_image()
{
    update_frame();
    if (threshold_value <= 0) {
        // Adaptive threshold
        cv::adaptiveThreshold(frame_gray, frame_threshold, 1.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 3, 1);
    } else {
        // Use simple threshold for thresh values: 0 < thresh < 255
        cv::threshold(frame_gray, frame_threshold, threshold_value, threshold_max_value, threshold_simple_type);
    }
    return mat_to_image(frame_threshold);
}

void CVCamera::flip(bool flip_lr, bool flip_ud)
{
    this->flip_lr = flip_lr;
    this->flip_ud = flip_ud;
}

void CVCamera::update_threshold_values(double thres_val, double thres_max_val)
{
    this->threshold_value = thres_val;
    this->threshold_max_value = thres_max_val;
}

void CVCamera::update_thres_type(int type)
{
    this->threshold_simple_type = type;
}

double CVCamera::get_framerate() const
{
    return framerate;
}