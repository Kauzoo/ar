/*
Contribution credits:
Leonard Keil <leonard.keil@tum.de>
Jannik Waldraff <jannik.waldraff@tum.de>
Sandro Weber <sandro.weber@tum.de>
*/

#ifndef CV_CAMERA_H
#define CV_CAMERA_H

#include <stdio.h>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/classes/image.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

struct StripDimensions {
    int stripLength;
    int nStop;
    int nStart;
    cv::Point2f stripeVecX;
    cv::Point2f stripeVecY;
};


namespace godot {

class CVCamera : public RefCounted {
	GDCLASS(CVCamera, RefCounted)

private:
	cv::VideoCapture capture;
    uint64_t last_update_frame;
    cv::Mat frame_raw;
    cv::Mat frame_rgb;
    cv::Mat frame_gray;
    cv::Mat frame_threshold;
    bool flip_lr, flip_ud;
    bool video_file_playback;
    double framerate;

    double threshold_value;
    double threshold_max_value;
    // THRESH_BINARY = 0, THRESH_BINARY_INV = 1, THRESH_TRUNC = 2, THRESH_TOZERO = 3, THRESH_TOZERO_INV = 4, THRESH_MASK = 7,   cv::THRESH_OTSU = 8 , cv::THRESH_TRIANGLE = 16 , cv::THRESH_DRYRUN = 128 
    int threshold_simple_type;
    // cv::ADAPTIVE_THRESH_MEAN_C = 0, cv::ADAPTIVE_THRESH_GAUSSIAN_C = 1
    int threshold_adaptive_type;
    int threshold_blocksize;    // adaptive only
    double threshold_c;     // adaptive only

    void update_frame();
    Ref<Image> mat_to_image(cv::Mat mat);

    // Helpers
    bool is_valid_threshold_type(int type);

protected:
    static void _bind_methods();

public:
	CVCamera();
	~CVCamera();

    void open(int device, int width, int height);
    void open_file(String path);
    void close();
    Ref<Image> get_image();
    Ref<Image> get_greyscale_image();
    Ref<Image> get_threshold_image();
    void flip(bool flip_lr, bool flip_ud);
    double get_framerate() const;
    void update_threshold_value(double thres_val);
    void update_threshold_max_value(double max_val);
    void update_thres_type(int type);
    void update_threshold_adaptive_type(int type);
    void update_threshold_blocksize(float blocksize);
    void update_threshold_c(double c);
};

} //namespace godot

#endif